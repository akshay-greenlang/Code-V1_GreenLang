# Contributing to GL-016 Waterguard

Thank you for your interest in contributing to GL-016 Waterguard! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Safety-Critical Code Guidelines](#safety-critical-code-guidelines)
- [Release Process](#release-process)

---

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

### Our Standards

- Be respectful and inclusive
- Focus on constructive feedback
- Accept responsibility for mistakes
- Prioritize safety in all contributions
- Maintain professional communication

---

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- Python 3.10 or higher (< 3.13)
- Git
- Docker and Docker Compose (for integration tests)
- PostgreSQL 14+ (or use Docker)
- Redis 7+ (or use Docker)

### Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/gl-016-waterguard.git
cd gl-016-waterguard

# Add upstream remote
git remote add upstream https://github.com/greenlang/gl-016-waterguard.git
```

---

## Development Setup

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
.\venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### 3. Set Up Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit with your local settings
# Required variables:
# - WATERGUARD_DB_URL
# - WATERGUARD_REDIS_URL
```

### 4. Start Local Services

```bash
# Start PostgreSQL and Redis with Docker
docker-compose -f docker-compose.dev.yml up -d

# Run database migrations
alembic upgrade head

# Seed test data (optional)
python scripts/seed_test_data.py
```

### 5. Verify Setup

```bash
# Run tests to verify setup
pytest tests/unit/ -v

# Start development server
uvicorn api.main:app --reload --port 8080

# Access API documentation
# http://localhost:8080/docs
```

---

## Development Workflow

### Branch Naming Convention

Use the following prefixes for branch names:

| Prefix | Purpose |
|--------|---------|
| `feature/` | New features |
| `bugfix/` | Bug fixes |
| `hotfix/` | Critical production fixes |
| `safety/` | Safety-critical changes |
| `docs/` | Documentation updates |
| `refactor/` | Code refactoring |
| `test/` | Test additions/improvements |

Example:
```bash
git checkout -b feature/improve-coc-calculation
```

### Commit Message Format

Follow the Conventional Commits specification:

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting (no code change)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance
- `safety`: Safety-critical change

Examples:
```bash
git commit -m "feat(optimization): add multi-objective CoC optimization"
git commit -m "fix(safety): correct rate limiter overflow handling"
git commit -m "safety(interlocks): add high-silica trip logic"
```

### Keeping Your Fork Updated

```bash
# Fetch upstream changes
git fetch upstream

# Rebase your branch on upstream/main
git rebase upstream/main

# Push to your fork
git push origin feature/your-feature
```

---

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications. Use the following tools:

```bash
# Format code with Black
black .

# Sort imports with isort
isort .

# Lint with Ruff
ruff check .

# Type check with mypy
mypy .
```

### Configuration Files

`.ruff.toml`:
```toml
line-length = 100
target-version = "py310"

[lint]
select = ["E", "F", "W", "I", "N", "D", "UP", "B", "C4", "SIM"]
ignore = ["D100", "D104"]
```

`pyproject.toml` (Black config):
```toml
[tool.black]
line-length = 100
target-version = ["py310", "py311", "py312"]
```

### Code Organization

```
gl-016-waterguard/
├── api/                    # API layer (REST, GraphQL, gRPC)
│   ├── routes/            # Route handlers
│   ├── schemas/           # Request/response schemas
│   └── middleware/        # API middleware
├── core/                   # Core business logic
│   ├── config.py          # Configuration management
│   ├── schemas.py         # Domain models
│   ├── handlers.py        # Event handlers
│   └── coordinators.py    # Business coordinators
├── calculators/           # Calculation engines
│   ├── coc.py            # Cycles of concentration
│   ├── mass_balance.py   # Mass balance calculations
│   └── thermodynamics.py # Thermodynamic calculations
├── control/               # Control logic
│   ├── blowdown.py       # Blowdown controller
│   ├── dosing.py         # Dosing controller
│   └── rate_limiter.py   # Rate limiting
├── optimization/          # Optimization engine
│   ├── optimizer.py      # CVXPY optimizer
│   └── cost_functions.py # Cost function definitions
├── safety/                # Safety-critical code
│   ├── interlocks.py     # Safety interlocks
│   ├── gates.py          # Safety gates
│   └── watchdog.py       # Watchdog timer
├── explainability/        # Explainability layer
│   ├── shap_analyzer.py  # SHAP analysis
│   └── lime_explainer.py # LIME explanations
├── audit/                 # Audit trail
│   ├── logger.py         # Event logging
│   └── provenance.py     # Hash chain provenance
├── tests/                 # Test suites
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── safety/           # Safety verification tests
└── docs/                  # Documentation
```

### Documentation Standards

All public functions, classes, and modules must have docstrings:

```python
def calculate_coc(
    blowdown_conductivity: float,
    makeup_conductivity: float,
) -> CoCResult:
    """
    Calculate Cycles of Concentration (CoC).

    The CoC represents the ratio of dissolved solids in the blowdown
    water to the dissolved solids in the makeup water.

    Args:
        blowdown_conductivity: Blowdown water conductivity (umho/cm)
        makeup_conductivity: Makeup water conductivity (umho/cm)

    Returns:
        CoCResult containing the calculated CoC and metadata

    Raises:
        ValueError: If makeup conductivity is zero or negative
        ChemistryError: If calculated CoC is outside valid range

    Example:
        >>> result = calculate_coc(3500.0, 500.0)
        >>> print(result.coc)
        7.0

    Note:
        This function is deterministic and produces reproducible results.
    """
    if makeup_conductivity <= 0:
        raise ValueError("Makeup conductivity must be positive")

    coc = blowdown_conductivity / makeup_conductivity

    if not (1.0 <= coc <= 20.0):
        raise ChemistryError(f"CoC {coc} outside valid range [1, 20]")

    return CoCResult(
        coc=coc,
        blowdown_conductivity=blowdown_conductivity,
        makeup_conductivity=makeup_conductivity,
        timestamp=datetime.utcnow(),
    )
```

---

## Testing Requirements

### Test Coverage Requirements

| Category | Minimum Coverage |
|----------|------------------|
| Core Logic | 90% |
| Safety Code | 100% |
| API Endpoints | 85% |
| Calculators | 95% |
| Overall | 80% |

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/safety/

# Run tests in parallel
pytest -n auto

# Run with verbose output
pytest -v

# Run a specific test
pytest tests/unit/test_coc.py::test_calculate_coc
```

### Test Naming Convention

```python
def test_<function_name>_<scenario>_<expected_result>():
    """Test docstring describing the test case."""
    pass

# Examples:
def test_calculate_coc_normal_input_returns_correct_value():
    """Test CoC calculation with normal input values."""
    pass

def test_calculate_coc_zero_makeup_raises_value_error():
    """Test that zero makeup conductivity raises ValueError."""
    pass

def test_safety_interlock_high_conductivity_triggers_trip():
    """Test high conductivity interlock triggers safety trip."""
    pass
```

### Safety Test Requirements

All safety-critical code must have:

1. **Positive Tests**: Verify correct behavior under normal conditions
2. **Negative Tests**: Verify error handling
3. **Boundary Tests**: Test at limit values
4. **Failure Mode Tests**: Verify fail-safe behavior
5. **Race Condition Tests**: Verify thread safety

```python
class TestHighConductivityInterlock:
    """Test suite for high conductivity safety interlock."""

    def test_normal_conductivity_no_trip(self):
        """Verify no trip at normal conductivity."""
        pass

    def test_alarm_threshold_triggers_warning(self):
        """Verify warning at alarm threshold."""
        pass

    def test_trip_threshold_triggers_interlock(self):
        """Verify interlock triggers at trip threshold."""
        pass

    def test_boundary_just_below_trip(self):
        """Verify no trip at boundary just below threshold."""
        pass

    def test_boundary_just_above_trip(self):
        """Verify trip at boundary just above threshold."""
        pass

    def test_sensor_failure_triggers_failsafe(self):
        """Verify fail-safe on sensor failure."""
        pass
```

---

## Documentation

### Documentation Requirements

All contributions should include appropriate documentation:

| Change Type | Documentation Required |
|-------------|------------------------|
| New Feature | README update, API docs, user guide |
| Bug Fix | CHANGELOG entry |
| API Change | OpenAPI spec update, migration guide |
| Safety Change | Safety analysis document |
| Configuration | Configuration guide update |

### Building Documentation

```bash
# Build MkDocs documentation
mkdocs build

# Serve documentation locally
mkdocs serve

# View at http://localhost:8000
```

---

## Pull Request Process

### Before Submitting

1. **Sync with upstream**: Rebase on latest main branch
2. **Run tests**: Ensure all tests pass
3. **Check coverage**: Verify coverage requirements are met
4. **Lint code**: Run all linters without errors
5. **Update docs**: Add/update documentation as needed
6. **Update CHANGELOG**: Add entry for your changes

### PR Template

```markdown
## Description
<!-- Describe your changes -->

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Safety-critical change
- [ ] Documentation update

## Testing
<!-- Describe testing performed -->
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Safety tests added (if applicable)
- [ ] All tests passing

## Safety Considerations
<!-- For safety-critical changes -->
- [ ] Safety analysis completed
- [ ] Fail-safe behavior verified
- [ ] Reviewed by safety engineer

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed code
- [ ] Documentation updated
- [ ] CHANGELOG updated
- [ ] No new warnings
```

### Review Process

1. **Automated Checks**: CI must pass
2. **Code Review**: At least 1 approval required
3. **Safety Review**: Required for safety-critical changes
4. **Documentation Review**: Verify docs are complete
5. **Final Approval**: Maintainer approval

---

## Safety-Critical Code Guidelines

### Identifying Safety-Critical Code

Code is safety-critical if it:

- Controls actuators (valves, pumps)
- Implements safety interlocks
- Affects operating modes
- Handles safety-related data

### Safety Code Requirements

1. **100% Test Coverage**: All safety code must have complete test coverage
2. **Defensive Programming**: Validate all inputs, handle all errors
3. **Fail-Safe Defaults**: Always default to safe state
4. **Logging**: Log all safety-related events
5. **Review**: Requires safety engineer review

### Safety Code Example

```python
class BlowdownValveController:
    """
    Controller for blowdown valve.

    SAFETY-CRITICAL: This class controls physical actuators.
    All changes require safety review.
    """

    # Safety constraints
    MAX_RATE_PERCENT_PER_SECOND = 10.0
    FAIL_SAFE_POSITION = 50.0  # Partially open

    def set_position(self, target_percent: float) -> ControlResult:
        """
        Set blowdown valve position.

        SAFETY-CRITICAL: Rate-limited to prevent water hammer.

        Args:
            target_percent: Target valve position (0-100%)

        Returns:
            ControlResult with actual position achieved

        Safety:
            - Input clamped to 0-100%
            - Rate limited to MAX_RATE_PERCENT_PER_SECOND
            - Fails to FAIL_SAFE_POSITION on error
        """
        # Validate input
        target_percent = max(0.0, min(100.0, target_percent))

        try:
            # Apply rate limiting
            current = self.get_current_position()
            rate_limited = self._apply_rate_limit(current, target_percent)

            # Execute command
            result = self._send_command(rate_limited)

            # Verify result
            if not self._verify_position(rate_limited):
                self._log_safety_event("position_verification_failed")
                return self._enter_safe_state()

            return result

        except Exception as e:
            self._log_safety_event("valve_control_error", error=str(e))
            return self._enter_safe_state()

    def _enter_safe_state(self) -> ControlResult:
        """Enter fail-safe state."""
        self._send_command(self.FAIL_SAFE_POSITION)
        return ControlResult(
            position=self.FAIL_SAFE_POSITION,
            status=ControlStatus.FAIL_SAFE,
        )
```

---

## Release Process

### Version Numbering

We follow Semantic Versioning (SemVer):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. [ ] All tests passing
2. [ ] Coverage requirements met
3. [ ] Documentation updated
4. [ ] CHANGELOG updated
5. [ ] Version bumped
6. [ ] Security scan passed
7. [ ] Safety review completed (if applicable)
8. [ ] Release notes prepared
9. [ ] Docker image built and tested
10. [ ] Helm chart updated

---

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Security Issues**: Email security@greenlang.io
- **Slack**: #gl-016-waterguard

---

## Recognition

Contributors are recognized in:

- CONTRIBUTORS.md file
- Release notes
- Documentation credits

Thank you for contributing to GL-016 Waterguard!
