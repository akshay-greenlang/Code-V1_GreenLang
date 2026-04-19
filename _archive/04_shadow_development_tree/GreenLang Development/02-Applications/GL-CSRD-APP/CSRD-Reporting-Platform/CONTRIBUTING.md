# Contributing to CSRD/ESRS Digital Reporting Platform

Thank you for your interest in contributing to the CSRD/ESRS Digital Reporting Platform! This document provides guidelines and instructions for contributing to this project.

---

## Table of Contents

1. [Welcome](#welcome)
2. [Code of Conduct](#code-of-conduct)
3. [How to Contribute](#how-to-contribute)
4. [Development Setup](#development-setup)
5. [Coding Standards](#coding-standards)
6. [Testing Requirements](#testing-requirements)
7. [Pull Request Process](#pull-request-process)
8. [Release Process](#release-process)
9. [Community Guidelines](#community-guidelines)
10. [Getting Help](#getting-help)

---

## Welcome

The CSRD/ESRS Digital Reporting Platform is an open-source solution for EU sustainability reporting compliance. We welcome contributions from:

- **Sustainability professionals** - Domain expertise on ESRS standards
- **Developers** - Code improvements, bug fixes, new features
- **Data scientists** - Enhanced algorithms and calculations
- **Documentation writers** - User guides, tutorials, translations
- **Compliance experts** - Regulatory guidance and validation
- **Testers** - Bug reports, QA testing, edge case identification

**Project Goals:**
- 100% calculation accuracy (zero-hallucination guarantee)
- Complete ESRS compliance (1,082 data points)
- Production-ready code quality
- Comprehensive documentation
- Welcoming community

---

## Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of:
- Age, body size, disability, ethnicity, gender identity and expression
- Level of experience, education, socio-economic status
- Nationality, personal appearance, race, religion
- Sexual identity and orientation

### Our Standards

**Positive behaviors:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behaviors:**
- Trolling, insulting/derogatory comments, personal or political attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported to:
- **Email**: conduct@greenlang.io
- **Response time**: Within 24 hours

All complaints will be reviewed and investigated. Project maintainers are obligated to maintain confidentiality with regard to the reporter of an incident.

### Attribution

This Code of Conduct is adapted from the [Contributor Covenant](https://www.contributor-covenant.org/version/2/0/code_of_conduct.html), version 2.0.

---

## How to Contribute

### Types of Contributions

#### 1. Bug Reports

**Before submitting a bug:**
- Check existing issues to avoid duplicates
- Test with the latest version
- Gather relevant information (version, OS, error messages)

**Create a bug report with:**
- **Title**: Clear, descriptive summary
- **Environment**: OS, Python version, package version
- **Steps to Reproduce**: Minimal reproducible example
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Error Messages**: Full stack traces
- **Additional Context**: Screenshots, logs, data samples (anonymized)

**Bug Report Template:**
```markdown
## Bug Description
[Clear description of the bug]

## Environment
- OS: [e.g., Ubuntu 22.04, Windows 11, macOS 13]
- Python Version: [e.g., 3.11.5]
- Package Version: [e.g., 1.0.0]

## Steps to Reproduce
1. [First step]
2. [Second step]
3. [...]

## Expected Behavior
[What you expected to happen]

## Actual Behavior
[What actually happened]

## Error Messages
```
[Paste error messages and stack traces here]
```

## Additional Context
[Screenshots, logs, etc.]
```

#### 2. Feature Requests

**Before proposing a feature:**
- Check if it already exists or is planned
- Consider if it fits the project scope (CSRD/ESRS compliance)
- Think about implementation complexity

**Create a feature request with:**
- **Use Case**: Why is this feature needed?
- **Proposed Solution**: How should it work?
- **Alternatives**: Other approaches considered
- **ESRS Relevance**: How does it support CSRD compliance?
- **Priority**: Low/Medium/High

**Feature Request Template:**
```markdown
## Feature Description
[Clear description of the proposed feature]

## Use Case
[Who needs this feature and why?]

## Proposed Solution
[How should this feature work?]

## Alternatives Considered
[Other approaches you've thought about]

## ESRS Relevance
[How does this support CSRD compliance?]

## Priority
[Low / Medium / High]
```

#### 3. Code Contributions

**Areas where we welcome code contributions:**

**High Priority:**
- Bug fixes
- Test coverage improvements
- Performance optimizations
- ESRS data point additions
- Documentation improvements

**Medium Priority:**
- New agent features
- Enhanced validation rules
- Additional language support
- Improved error messages

**Low Priority (Requires Discussion First):**
- Major architectural changes
- New dependencies
- Breaking API changes

#### 4. Documentation Improvements

**Documentation needs:**
- User guides and tutorials
- API documentation
- ESRS implementation examples
- Multilingual translations (DE, FR, ES)
- Video tutorials
- FAQ additions

#### 5. Testing & QA

**Testing contributions:**
- New test cases
- Edge case identification
- Performance benchmarking
- Integration testing
- User acceptance testing
- Real-world data validation (anonymized)

---

## Development Setup

### Prerequisites

- **Python**: 3.11 or higher
- **Git**: Latest version
- **PostgreSQL**: 14+ (for database-dependent features)
- **OS**: Linux, macOS, or Windows

### Initial Setup

```bash
# 1. Fork the repository on GitHub
# Click "Fork" button at https://github.com/akshay-greenlang/Code-V1_GreenLang

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/Code-V1_GreenLang.git
cd Code-V1_GreenLang/GL-CSRD-APP/CSRD-Reporting-Platform

# 3. Add upstream remote
git remote add upstream https://github.com/akshay-greenlang/Code-V1_GreenLang.git

# 4. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 5. Upgrade pip
pip install --upgrade pip

# 6. Install dependencies
pip install -r requirements.txt

# 7. Install development dependencies
pip install -r requirements-dev.txt  # If available, or install manually:
pip install pytest pytest-cov black ruff mypy ipython

# 8. Set up environment variables
cp .env.example .env
# Edit .env with your configuration (API keys, etc.)

# 9. Run tests to verify setup
pytest tests/

# 10. Create a feature branch
git checkout -b feature/your-feature-name
```

### Development Workflow

```bash
# 1. Sync with upstream before starting work
git fetch upstream
git merge upstream/master

# 2. Make your changes
# Edit files, add features, fix bugs...

# 3. Run tests frequently
pytest tests/

# 4. Run linting
ruff check .
black .

# 5. Run type checking
mypy agents/

# 6. Commit your changes (atomic commits)
git add .
git commit -m "feat: add ESRS E2 pollution metrics"

# 7. Push to your fork
git push origin feature/your-feature-name

# 8. Create Pull Request on GitHub
# Go to your fork on GitHub and click "New Pull Request"
```

---

## Coding Standards

### Python Style Guide

**Follow PEP 8** with these specific guidelines:

#### Code Formatting
- **Line length**: 100 characters max (not 79)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Double quotes for strings
- **Imports**: Organized (stdlib, third-party, local)
- **Formatter**: Black (with line-length=100)

```bash
# Format code with Black
black --line-length 100 agents/
```

#### Naming Conventions
- **Variables**: `snake_case`
- **Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: `_leading_underscore`

```python
# Good
class CalculatorAgent:
    MAX_RETRY_COUNT = 3

    def calculate_ghg_emissions(self, scope: int) -> float:
        emission_factor = self._get_emission_factor(scope)
        return emission_factor * activity_data

# Bad
class calculatorAgent:  # Wrong case
    maxRetryCount = 3  # Wrong case

    def CalculateGHGEmissions(self, Scope):  # Wrong case
        EmissionFactor = self.GetEmissionFactor(Scope)  # Wrong case
        return EmissionFactor * ActivityData
```

#### Type Hints
**Required for all function signatures:**

```python
from typing import List, Dict, Optional
from pydantic import BaseModel

# Good
def validate_esg_data(
    data: List[Dict[str, any]],
    schema: Dict[str, any],
    strict: bool = True
) -> tuple[bool, List[str]]:
    """Validate ESG data against schema."""
    errors: List[str] = []
    # ... implementation
    return len(errors) == 0, errors

# Bad (missing type hints)
def validate_esg_data(data, schema, strict=True):
    errors = []
    return len(errors) == 0, errors
```

#### Docstrings
**Required for all public functions and classes:**

```python
def calculate_scope_1_emissions(
    activity_data: float,
    emission_factor: float,
    unit: str = "tCO2e"
) -> float:
    """
    Calculate Scope 1 GHG emissions per GHG Protocol.

    Args:
        activity_data: Activity data value (e.g., fuel consumption)
        emission_factor: Emission factor from GHG Protocol database
        unit: Output unit (default: tCO2e)

    Returns:
        Total Scope 1 emissions in specified unit

    Raises:
        ValueError: If activity_data or emission_factor is negative

    Example:
        >>> calculate_scope_1_emissions(1000, 2.5)
        2500.0
    """
    if activity_data < 0 or emission_factor < 0:
        raise ValueError("Activity data and emission factor must be non-negative")

    return activity_data * emission_factor
```

#### Error Handling
**Use specific exceptions, not bare `except`:**

```python
# Good
try:
    result = calculate_metric(data)
except ValueError as e:
    logger.error(f"Invalid data: {e}")
    raise
except KeyError as e:
    logger.error(f"Missing required field: {e}")
    raise

# Bad
try:
    result = calculate_metric(data)
except:  # Too broad
    pass  # Silently swallowing errors
```

#### Logging
**Use structured logging:**

```python
import logging

logger = logging.getLogger(__name__)

# Good
logger.info("Calculating metrics", extra={
    "company_id": company_id,
    "metric_count": len(metrics),
    "esrs_standard": "E1"
})

# Bad
print(f"Calculating metrics for {company_id}")  # Don't use print
```

### Zero-Hallucination Principle

**CRITICAL: Never use LLM for numeric calculations or compliance decisions.**

```python
# ✅ GOOD: Deterministic calculation
def calculate_energy_intensity(energy_kwh: float, revenue_eur: float) -> float:
    """100% deterministic, zero-hallucination."""
    return energy_kwh / revenue_eur

# ❌ BAD: LLM-based calculation (FORBIDDEN)
def calculate_energy_intensity_ai(company_data: str) -> float:
    """This violates zero-hallucination principle!"""
    prompt = f"Calculate energy intensity for: {company_data}"
    return llm.generate(prompt)  # NEVER DO THIS
```

**AI Usage Restrictions:**
- ✅ **Allowed**: Materiality assessment (with human review)
- ✅ **Allowed**: Narrative generation (with human review)
- ❌ **Forbidden**: Numeric calculations
- ❌ **Forbidden**: Compliance decisions
- ❌ **Forbidden**: Data validation logic

---

## Testing Requirements

### Test Coverage Standards

**Minimum coverage requirements:**
- **Overall**: 85% line coverage
- **Core agents**: 95% line coverage
- **Critical calculations**: 100% line coverage

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_calculator_agent.py

# Run with coverage report
pytest --cov=agents --cov-report=html tests/

# Run only fast tests (skip slow integration tests)
pytest -m "not slow" tests/

# Run only agent tests
pytest tests/test_*_agent.py
```

### Writing Tests

**Test file structure:**
```python
# tests/test_calculator_agent.py
import pytest
from agents.calculator_agent import CalculatorAgent

class TestCalculatorAgent:
    """Test suite for CalculatorAgent."""

    @pytest.fixture
    def calculator_agent(self):
        """Fixture to create CalculatorAgent instance."""
        return CalculatorAgent()

    def test_scope_1_emissions_calculation(self, calculator_agent):
        """Test Scope 1 GHG emissions calculation."""
        # Arrange
        activity_data = 1000.0  # Liters of diesel
        emission_factor = 2.68  # kgCO2e/liter

        # Act
        result = calculator_agent.calculate_scope_1_emissions(
            activity_data, emission_factor
        )

        # Assert
        assert result == 2680.0
        assert isinstance(result, float)

    def test_invalid_negative_emission_factor(self, calculator_agent):
        """Test that negative emission factor raises ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            calculator_agent.calculate_scope_1_emissions(1000, -2.68)
```

**Test categories:**

```python
# Mark tests by category
@pytest.mark.unit
def test_individual_function():
    """Unit test for single function."""
    pass

@pytest.mark.integration
def test_agent_pipeline():
    """Integration test for agent interactions."""
    pass

@pytest.mark.slow
def test_large_dataset():
    """Slow test (skipped in fast mode)."""
    pass

@pytest.mark.esrs_e1
def test_climate_metrics():
    """Test ESRS E1 (Climate) metrics."""
    pass
```

### Test Data

**Use fixtures for test data:**

```python
# tests/conftest.py
import pytest

@pytest.fixture
def sample_esg_data():
    """Sample ESG data for testing."""
    return {
        "E1-1": {"value": 12500, "unit": "tCO2e"},
        "E1-2": {"value": 8300, "unit": "tCO2e"},
    }

@pytest.fixture
def sample_company_profile():
    """Sample company profile for testing."""
    return {
        "legal_name": "Test Company B.V.",
        "lei_code": "549300ABC123DEF456GH",
        "country": "NL",
    }
```

---

## Pull Request Process

### Before Submitting a PR

**Checklist:**
- [ ] Code follows style guidelines (Black formatted, Ruff clean)
- [ ] Type hints added for all functions
- [ ] Docstrings added for all public functions
- [ ] Tests added for new features
- [ ] All tests passing (`pytest tests/`)
- [ ] Coverage maintained or improved
- [ ] Documentation updated (if needed)
- [ ] CHANGELOG.md updated (if user-facing change)
- [ ] No merge conflicts with master branch
- [ ] Commit messages follow conventions (see below)

### Commit Message Conventions

**Format:** `<type>(<scope>): <subject>`

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, no code change
- `refactor`: Code restructuring, no behavior change
- `test`: Adding tests
- `chore`: Build, dependencies, etc.

**Examples:**
```bash
feat(calculator): add ESRS E2 pollution metrics
fix(intake): handle missing data in CSV parser
docs(readme): update installation instructions
test(materiality): add edge case tests for impact scoring
```

### PR Title and Description

**PR Title Format:**
```
<type>: <clear description>

Examples:
feat: Add ESRS E2 pollution metrics calculation
fix: Resolve materiality scoring edge case
docs: Update ESRS implementation guide
```

**PR Description Template:**
```markdown
## Description
[What does this PR do?]

## Motivation
[Why is this change needed?]

## Changes Made
- [Change 1]
- [Change 2]
- [...]

## Testing
[How was this tested?]

## Screenshots (if applicable)
[Add screenshots for UI changes]

## Related Issues
Closes #123
Related to #456

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] All tests passing
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests, linting, type checking
2. **Peer Review**: At least one maintainer reviews code
3. **Feedback**: Address review comments, push updates
4. **Approval**: Maintainer approves PR
5. **Merge**: Maintainer merges to master

**Review timeline:**
- Small PRs (<100 lines): 1-2 days
- Medium PRs (100-500 lines): 3-5 days
- Large PRs (>500 lines): 1-2 weeks

**Tips for faster review:**
- Keep PRs small and focused
- Write clear descriptions
- Respond promptly to feedback
- Don't force-push after review starts

---

## Release Process

### Versioning

**We follow Semantic Versioning (SemVer):**
- **Major** (1.0.0 → 2.0.0): Breaking changes
- **Minor** (1.0.0 → 1.1.0): New features, backwards compatible
- **Patch** (1.0.0 → 1.0.1): Bug fixes, backwards compatible

### Release Workflow

**Release Manager responsibilities:**

1. **Prepare Release Branch**
   ```bash
   git checkout -b release/v1.1.0
   ```

2. **Update Version Numbers**
   - `setup.py`: Update `version`
   - `pack.yaml`: Update `version`
   - `gl.yaml`: Update `version`
   - `__init__.py`: Update `__version__`

3. **Update CHANGELOG.md**
   - Move "Unreleased" items to new version section
   - Add release date
   - Review all changes

4. **Run Full Test Suite**
   ```bash
   pytest tests/
   pytest --cov=agents --cov-report=html tests/
   ```

5. **Security Scan**
   ```bash
   safety check
   bandit -r agents/
   ```

6. **Build Package**
   ```bash
   python setup.py sdist bdist_wheel
   ```

7. **Create Release PR**
   - Title: "Release v1.1.0"
   - Description: Copy CHANGELOG for this version
   - Get approval from maintainers

8. **Merge and Tag**
   ```bash
   git checkout master
   git merge release/v1.1.0
   git tag -a v1.1.0 -m "Release version 1.1.0"
   git push origin master --tags
   ```

9. **Publish Release**
   - Create GitHub Release with CHANGELOG
   - Attach built packages
   - Publish to PyPI (if applicable)

10. **Announce Release**
    - Email announcement to mailing list
    - Social media announcement
    - Update documentation site

---

## Community Guidelines

### Communication Channels

**GitHub Issues**: Bug reports, feature requests
**GitHub Discussions**: Questions, ideas, general discussion
**Email**: csrd@greenlang.io (project inquiries)
**Twitter**: @greenlang_io (announcements)

### Asking Questions

**Before asking:**
- Search existing issues and discussions
- Read documentation thoroughly
- Try to solve it yourself (learning opportunity!)

**When asking:**
- Be specific and provide context
- Include code samples and error messages
- Show what you've tried
- Be patient waiting for responses

### Providing Feedback

**We value all feedback:**
- Feature suggestions
- Bug reports
- Documentation improvements
- Usability issues
- Performance problems

**Be constructive:**
- Explain the problem and impact
- Suggest potential solutions
- Provide examples and data
- Be respectful and professional

### Recognition

**Contributors are recognized:**
- In CHANGELOG.md
- In release notes
- On project website (if applicable)
- In academic papers (if applicable)

---

## Getting Help

### Documentation
- **README.md**: Project overview and quick start
- **docs/**: Comprehensive documentation
- **API Reference**: `docs/API_REFERENCE.md`
- **ESRS Guide**: `docs/ESRS_GUIDE.md`

### Support Channels
- **GitHub Issues**: Technical questions
- **GitHub Discussions**: General questions
- **Email**: csrd@greenlang.io

### Mentorship

**New contributors:**
- Look for issues tagged `good-first-issue`
- Ask for help in GitHub Discussions
- Request code review feedback
- Pair programming available (reach out!)

---

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

---

## Thank You!

Thank you for contributing to the CSRD/ESRS Digital Reporting Platform! Your contributions help companies worldwide achieve sustainability reporting compliance.

**Special Thanks To:**
- All our contributors
- The ESRS implementation community
- Open source projects we depend on
- Users providing feedback and bug reports

---

**Last Updated**: 2025-10-18
**Version**: 1.0.0
**Contact**: csrd@greenlang.io
