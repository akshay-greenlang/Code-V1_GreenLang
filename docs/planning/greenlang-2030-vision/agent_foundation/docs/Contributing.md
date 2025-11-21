# Contributing to GreenLang Agent Foundation

## Welcome Contributors!

Thank you for your interest in contributing to GreenLang! This document provides guidelines for contributing to the project.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contribution Workflow](#contribution-workflow)
5. [Coding Standards](#coding-standards)
6. [Testing Requirements](#testing-requirements)
7. [Documentation Guidelines](#documentation-guidelines)
8. [Pull Request Process](#pull-request-process)
9. [Community](#community)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors.

### Expected Behavior

- Be respectful and constructive
- Welcome diverse perspectives
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy toward others

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Personal or political attacks
- Public or private harassment
- Publishing others' private information

---

## Getting Started

### Find Something to Work On

1. **Good First Issues**: Look for issues labeled `good-first-issue`
2. **Help Wanted**: Check `help-wanted` label
3. **Documentation**: Improve docs (always appreciated!)
4. **Bug Fixes**: Fix reported bugs
5. **New Features**: Propose and implement features

### Before You Start

1. Check if issue exists
2. Comment on the issue to claim it
3. Discuss approach if implementing large feature
4. Fork the repository
5. Create a branch for your work

---

## Development Setup

### Prerequisites

```bash
# Required
- Python 3.11+
- Git
- Docker (for integration tests)

# Recommended
- VSCode or PyCharm
- PostgreSQL (local)
- Redis (local)
```

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/greenlang/agent-foundation.git
cd agent-foundation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Verify setup
pytest tests/
```

### Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your local settings
DATABASE_URL=postgresql://localhost/greenlang_dev
REDIS_URL=redis://localhost:6379
OPENAI_API_KEY=your-key-for-testing
```

---

## Contribution Workflow

### 1. Create a Branch

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Or bug fix branch
git checkout -b fix/bug-description
```

### 2. Make Changes

```bash
# Make your changes
# Follow coding standards
# Add tests
# Update documentation
```

### 3. Test Your Changes

```bash
# Run tests
pytest tests/

# Run specific tests
pytest tests/test_my_feature.py

# Run with coverage
pytest --cov=greenlang tests/

# Run linting
flake8 src/
black src/
mypy src/
```

### 4. Commit Changes

```bash
# Stage changes
git add .

# Commit with meaningful message
git commit -m "feat: add carbon calculation agent

- Implement GHG Protocol calculations
- Add Scope 1, 2, 3 support
- Include tests and documentation

Closes #123"
```

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

**Example:**
```
feat(memory): implement episodic memory system

Add episodic memory for storing agent experiences with
pattern extraction and replay capabilities.

Closes #456
```

### 5. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create Pull Request on GitHub
# Fill out the PR template
```

---

## Coding Standards

### Python Style Guide

Follow PEP 8 and our additional conventions:

```python
# Import order
import asyncio  # Standard library
import logging

import numpy as np  # Third-party
import pandas as pd

from greenlang.base import BaseAgent  # Local

# Type hints
def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process data and return result.

    Args:
        data: Input data dictionary

    Returns:
        Processed result dictionary

    Raises:
        ValidationError: If data is invalid
    """
    pass

# Docstrings (Google style)
class MyAgent(BaseAgent):
    """
    Agent for specific task.

    This agent performs...

    Attributes:
        config: Agent configuration
        memory: Memory manager instance

    Example:
        >>> agent = MyAgent(config)
        >>> result = agent.process(data)
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize agent.

        Args:
            config: Agent configuration object
        """
        super().__init__(config)

# Use descriptive names
def calculate_carbon_emissions(activity_data):  # Good
    pass

def calc(data):  # Bad
    pass

# Constants
MAX_RETRY_ATTEMPTS = 3  # Good
max = 3  # Bad
```

### Code Quality Tools

```bash
# Auto-formatting
black src/
isort src/

# Linting
flake8 src/
pylint src/

# Type checking
mypy src/

# Security scanning
bandit -r src/
```

---

## Testing Requirements

### Test Coverage

- Minimum 80% coverage for new code
- 100% coverage for critical paths
- Unit tests for all functions
- Integration tests for workflows
- E2E tests for user scenarios

### Writing Tests

```python
import pytest
from unittest.mock import AsyncMock

class TestMyFeature:
    """Test suite for my feature."""

    @pytest.fixture
    async def agent(self):
        """Create test agent."""
        config = AgentConfig(name="test")
        agent = MyAgent(config)
        await agent.initialize()
        return agent

    @pytest.mark.asyncio
    async def test_basic_functionality(self, agent):
        """Test basic functionality."""
        result = await agent.process({'data': 'test'})
        assert result['status'] == 'success'

    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        """Test error handling."""
        with pytest.raises(ValidationError):
            await agent.process({'invalid': 'data'})

    @pytest.mark.integration
    async def test_integration(self, agent):
        """Integration test."""
        # Test with real dependencies
        result = await agent.full_workflow()
        assert result['completed'] == True
```

---

## Documentation Guidelines

### Code Documentation

```python
def complex_function(param1: str, param2: int) -> Dict[str, Any]:
    """
    Brief description of what function does.

    More detailed explanation if needed. Can span
    multiple lines.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is invalid
        RuntimeError: When operation fails

    Example:
        >>> result = complex_function("test", 42)
        >>> print(result)
        {'status': 'success'}
    """
    pass
```

### User Documentation

- Write clear, concise tutorials
- Include code examples
- Add screenshots/diagrams
- Provide troubleshooting tips
- Link to related documentation

---

## Pull Request Process

### PR Checklist

Before submitting your PR, ensure:

- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] Commit messages are clear
- [ ] PR description is comprehensive
- [ ] No merge conflicts
- [ ] CI/CD passing
- [ ] Reviewed own code

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How was this tested?

## Checklist
- [ ] Tests added
- [ ] Documentation updated
- [ ] Code reviewed

## Related Issues
Closes #123
```

### Review Process

1. **Automated Checks**: CI/CD must pass
2. **Code Review**: At least one approval required
3. **Testing**: Reviewer tests functionality
4. **Discussion**: Address feedback
5. **Merge**: Once approved

---

## Community

### Communication Channels

- **GitHub Discussions**: https://github.com/greenlang/discussions
- **Discord**: https://discord.gg/greenlang
- **Twitter**: @GreenLangAI
- **Email**: community@greenlang.ai

### Weekly Calls

- **Time**: Tuesdays 10 AM PT
- **Link**: https://meet.greenlang.ai
- **Agenda**: Published in Discord

### Recognition

Contributors are recognized in:
- Release notes
- Contributors page
- Annual awards
- Conference speaking opportunities

---

## Questions?

- Check [FAQ](Getting_Started.md#faq)
- Ask in [Discord](https://discord.gg/greenlang)
- Email community@greenlang.ai

---

Thank you for contributing to GreenLang! 🌱

**Last Updated**: November 2024