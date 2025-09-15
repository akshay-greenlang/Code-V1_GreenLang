# Contributing to GreenLang

Thank you for your interest in contributing to GreenLang! We're building an open developer climate intelligence platform, and we welcome contributions from the community.

## 🌍 Our Mission

GreenLang aims to democratize access to climate intelligence and accelerate the global transition to net-zero emissions through open-source technology.

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- A GitHub account

### Development Setup

1. **Fork the repository**
   ```bash
   # Click "Fork" on https://github.com/greenlang/greenlang
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/greenlang.git
   cd greenlang
   ```

3. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install in development mode**
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

5. **Run tests to verify setup**
   ```bash
   pytest
   ```

## 📝 How to Contribute

### Types of Contributions

#### 🐛 Bug Reports
- Use the [Bug Report](https://github.com/greenlang/greenlang/issues/new?template=bug_report.md) template
- Include minimal reproducible example
- Specify GreenLang version and environment

#### ✨ Feature Requests
- Use the [Feature Request](https://github.com/greenlang/greenlang/issues/new?template=feature_request.md) template
- Explain the use case and benefits
- Consider implementation approach

#### 📊 Data Contributions
- **Emission Factors**: Add/update country emission factors with sources
- **Benchmarks**: Contribute regional building performance standards
- **Datasets**: Share validated climate data

#### 🌍 Translations
- Help translate documentation
- Localize CLI messages
- Add regional context

#### 📚 Documentation
- Fix typos and clarify explanations
- Add examples and tutorials
- Improve API documentation

#### 💻 Code Contributions
- Fix bugs
- Implement features
- Improve performance
- Add tests

### Development Workflow

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-number-description
   ```

2. **Make changes**
   - Write clean, documented code
   - Follow existing patterns
   - Add/update tests
   - Update documentation

3. **Run quality checks**
   ```bash
   # Run tests
   pytest
   
   # Check test coverage
   pytest --cov=greenlang --cov-report=html
   
   # Run linting
   ruff check greenlang/
   
   # Check types
   mypy greenlang/ --strict
   
   # Format code
   black greenlang/ tests/
   
   # Security scan
   bandit -r greenlang/
   ```

4. **Commit changes**
   ```bash
   git add .
   git commit -m "feat: add new emission factor for country X"
   ```
   
   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation
   - `test:` Tests
   - `refactor:` Code refactoring
   - `perf:` Performance improvement
   - `chore:` Maintenance

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

## 🧪 Testing Guidelines

### Test Requirements

- **All new code must have tests**
- **Maintain ≥95% coverage**
- **Pass all quality gates**

### Writing Tests

#### Unit Tests
```python
# tests/unit/test_your_feature.py
import pytest
from greenlang.your_module import your_function

def test_your_function():
    """Test description."""
    result = your_function(input_data)
    assert result == expected_output
```

#### Integration Tests
```python
# tests/integration/test_workflow_your_feature.py
@pytest.mark.integration
class TestYourFeature:
    def test_end_to_end(self, workflow_runner):
        """Test complete workflow."""
        result = workflow_runner.run(workflow, data)
        assert result['success'] is True
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific category
pytest -m integration
pytest -m unit

# Run with coverage
pytest --cov=greenlang --cov-fail-under=85

# Run in parallel
pytest -n auto
```

## 📋 Code Style Guide

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use type hints
- Document all public functions
- Keep functions focused and small

### Example Code

```python
from typing import Dict, List, Optional
from pydantic import BaseModel

class EmissionData(BaseModel):
    """Emission data model."""
    
    fuel_type: str
    value: float
    unit: str
    
def calculate_emissions(
    consumption: float,
    factor: float,
    unit: str = "kWh"
) -> Dict[str, float]:
    """
    Calculate emissions from consumption.
    
    Args:
        consumption: Energy consumption value
        factor: Emission factor (kgCO2e/unit)
        unit: Unit of consumption
        
    Returns:
        Dictionary with emissions data
        
    Example:
        >>> calculate_emissions(1000, 0.5)
        {"emissions_kg": 500.0}
    """
    emissions_kg = consumption * factor
    return {
        "emissions_kg": emissions_kg,
        "emissions_tons": emissions_kg / 1000
    }
```

### Data Contributions

When contributing emission factors or benchmarks:

```json
{
  "country": "XX",
  "electricity": {
    "value": 0.XXX,
    "unit": "kgCO2e/kWh",
    "source": "Government Agency Name",
    "year": 2024,
    "url": "https://source.url"
  }
}
```

## 🔄 Pull Request Process

### Before Submitting

- [ ] Tests pass locally
- [ ] Coverage ≥95%
- [ ] Code formatted with `black`
- [ ] No linting errors
- [ ] Documentation updated
- [ ] Commit messages follow convention

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Performance improvement

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests passing

## Checklist
- [ ] Code follows style guide
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes
```

### Review Process

1. Automated checks run
2. Code review by maintainers
3. Address feedback
4. Approval and merge

## 🏗️ Architecture Decisions

### Adding New Agents

1. Inherit from `BaseAgent`
2. Implement `execute()` method
3. Add contract validation
4. Write comprehensive tests
5. Document in agent registry

### Adding New Workflows

1. Create YAML definition
2. Define inputs/outputs schema
3. Add integration tests
4. Document usage

### Adding Country Data

1. Research official sources
2. Add to `global_emission_factors.json`
3. Include provenance information
4. Add tests for new country
5. Update documentation

## 🌟 Recognition

### Contributors

All contributors are recognized in:
- [CONTRIBUTORS.md](CONTRIBUTORS.md)
- GitHub contributors page
- Release notes

### Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

## 📮 Communication

### Channels

- **GitHub Issues**: Bug reports and features
- **GitHub Discussions**: General discussions
- **Discord**: [Join our Discord](https://discord.gg/greenlang)
- **Email**: contribute@greenlang.ai

### Getting Help

- Check [documentation](GREENLANG_DOCUMENTATION.md)
- Search existing issues
- Ask in Discord #help channel
- Email maintainers

## 🎯 Priority Areas

Current priority contributions:

1. **Emission Factors**: More countries needed
2. **Scope 3 Emissions**: Supply chain calculations
3. **Real-time Grid APIs**: Integration with grid operators
4. **ML Models**: Consumption prediction
5. **Visualizations**: Dashboard components
6. **Mobile SDKs**: iOS/Android libraries

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You!

Thank you for contributing to GreenLang and helping build a sustainable future! 🌍💚

---

**Questions?** Feel free to reach out to the maintainers or ask in our Discord community.