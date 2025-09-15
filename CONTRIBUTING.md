# Contributing to GreenLang

Thank you for your interest in contributing to GreenLang! We welcome contributions from the community.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/greenlang.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `pytest tests/`
6. Commit your changes: `git commit -m "Add feature: description"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Development Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Style

- We use Black for Python code formatting
- We use isort for import sorting
- Run `pre-commit run --all-files` before committing

## Testing

- Write tests for any new functionality
- Ensure all tests pass: `pytest tests/`
- Maintain or improve code coverage

## Pull Request Requirements

Before submitting a PR, ensure:

- [ ] All CI checks pass (CI, Pack Validation, Pipeline Validation)
- [ ] Tests are included for new functionality
- [ ] Documentation is updated if needed
- [ ] Code follows project style guidelines
- [ ] Commit messages are clear and descriptive

## Issue Labels

Look for issues labeled:
- `good first issue` - Great for newcomers
- `help wanted` - We need help with these
- `documentation` - Documentation improvements
- `bug` - Something isn't working
- `enhancement` - New feature or request

## Release Process

Releases are automated when a tag is pushed:

```bash
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

## Questions?

Feel free to open an issue for any questions or discussions.

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to maintain a welcoming and inclusive community.