# GreenLang Agent Factory

[![CI](https://github.com/greenlang/agent-factory/actions/workflows/ci.yml/badge.svg)](https://github.com/greenlang/agent-factory/actions/workflows/ci.yml)
[![CD](https://github.com/greenlang/agent-factory/actions/workflows/cd.yml/badge.svg)](https://github.com/greenlang/agent-factory/actions/workflows/cd.yml)
[![Agent Certification](https://github.com/greenlang/agent-factory/actions/workflows/agent-certification.yml/badge.svg)](https://github.com/greenlang/agent-factory/actions/workflows/agent-certification.yml)
[![Golden Tests](https://github.com/greenlang/agent-factory/actions/workflows/golden-tests.yml/badge.svg)](https://github.com/greenlang/agent-factory/actions/workflows/golden-tests.yml)

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![Code Coverage](https://codecov.io/gh/greenlang/agent-factory/branch/main/graph/badge.svg)](https://codecov.io/gh/greenlang/agent-factory)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat)](https://pycqa.github.io/isort/)
[![Linting: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue)](https://mypy-lang.org/)

[![Security: Bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

The **GreenLang Agent Factory** is a comprehensive system for designing, generating, evaluating, and operating advanced climate and industrial decarbonization agents at scale.

### North Star Vision

> **"We're building a factory that takes a high-level spec for a climate/industrial problem and generates the agent graph, code, prompts, tests, and evaluation suite - then certifies it against climate science and regulatory criteria."**

### Key Features

- **Agent SDK v1**: Type-safe `AgentSpecV2Base[InT, OutT]` interface with zero-hallucination guarantees
- **Agent Generator**: Automated code generation from YAML specifications
- **12-Dimension Certification**: Rigorous quality and compliance validation
- **Agent Registry**: Version-controlled agent lifecycle management
- **Production Runtime**: Kubernetes-native deployment with autoscaling

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/greenlang/agent-factory.git
cd agent-factory

# Install dependencies
make install

# Set up pre-commit hooks
make setup-hooks
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make coverage

# Run specific test suites
make test-unit
make test-integration
make test-golden
```

### Code Quality

```bash
# Run linting
make lint

# Auto-fix linting issues
make format

# Run security scans
make security

# Run all checks
make check
```

### Building and Deploying

```bash
# Build Docker image
make build

# Deploy to staging
make deploy-staging

# Deploy to production (requires confirmation)
make deploy-prod
```

### Agent Certification

```bash
# Run full certification
make certify

# Certify specific agent
make certify-agent AGENT=GL-001

# Run golden tests
make golden-tests
```

---

## Project Structure

```
GL-Agent-Factory/
├── .github/
│   └── workflows/
│       ├── ci.yml                    # Continuous Integration
│       ├── cd.yml                    # Continuous Deployment
│       ├── agent-certification.yml   # 12-Dimension Certification
│       ├── golden-tests.yml          # Determinism Tests
│       └── release.yml               # Release Pipeline
├── 00-foundation/                    # Vision & Strategy
├── 01-architecture/                  # System Design
├── 02-sdk/                           # Agent SDK v1
├── 03-generator/                     # Agent Generator
├── 04-evaluation/                    # Certification Framework
├── 05-registry/                      # Agent Registry
├── 06-teams/                         # Team Charters
├── 07-phases/                        # Implementation Roadmap
├── tests/
│   ├── unit/                         # Unit tests
│   ├── integration/                  # Integration tests
│   ├── e2e/                          # End-to-end tests
│   └── golden/                       # Golden tests
├── .pre-commit-config.yaml           # Pre-commit hooks
├── Makefile                          # CI/CD automation
├── pytest.ini                        # Pytest configuration
├── .coveragerc                       # Coverage configuration
├── tox.ini                           # Multi-Python testing
└── README.md                         # This file
```

---

## CI/CD Pipeline

### Continuous Integration (`ci.yml`)

Runs on all pushes and pull requests:

- **Lint**: Ruff, Black, isort, mypy
- **Test**: pytest with coverage (Python 3.10, 3.11, 3.12)
- **Security**: Bandit, Safety, pip-audit
- **Build**: Docker image with multi-arch support

### Continuous Deployment (`cd.yml`)

Runs on pushes to main/master:

1. Build and push Docker image
2. Deploy to staging
3. Run integration tests
4. Manual approval gate
5. Deploy to production
6. Smoke tests and rollback on failure

### Agent Certification (`agent-certification.yml`)

Runs when agent code changes:

- **12-Dimension Certification**:
  1. Specification Completeness
  2. Code Implementation
  3. Test Coverage (>85%)
  4. Deterministic AI Guarantees
  5. Documentation Completeness
  6. Compliance & Security
  7. Deployment Readiness
  8. Exit Bar Criteria
  9. Integration & Coordination
  10. Business Impact & Metrics
  11. Operational Excellence
  12. Continuous Improvement

### Golden Tests (`golden-tests.yml`)

Runs daily and on-demand:

- Cross-platform consistency (Ubuntu, Windows, macOS)
- Cross-version consistency (Python 3.10-3.12)
- Numerical precision verification
- Determinism checks (10 iterations)

### Release Pipeline (`release.yml`)

Triggered on version tags:

- Semantic versioning validation
- Changelog generation
- Python package build
- PyPI publishing
- Docker image tagging
- GitHub Release creation

---

## Development Workflow

### Pre-commit Hooks

```yaml
# Installed hooks
- pre-commit-hooks (file checks)
- black (formatting)
- isort (import sorting)
- ruff (linting)
- mypy (type checking)
- bandit (security)
- detect-secrets (secrets detection)
```

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add boiler efficiency calculator
fix: correct emission factor lookup
docs: update API documentation
chore: update dependencies
```

### Branch Strategy

- `main` / `master`: Production-ready code
- `develop`: Integration branch
- `feature/*`: Feature branches
- `fix/*`: Bug fix branches
- `release/*`: Release preparation

---

## Quality Standards

### Code Coverage

- **Minimum**: 85% overall coverage
- **Unit Tests**: >90% coverage
- **Integration Tests**: >80% coverage
- **Golden Tests**: 100% pass rate

### Performance Targets

| Metric | Target |
|--------|--------|
| P50 Latency | <2.0s |
| P95 Latency | <4.0s |
| P99 Latency | <6.0s |
| Cost per Analysis | <$0.15 |
| Success Rate | >99% |

### Security Requirements

- Zero P0/P1 vulnerabilities
- No hardcoded secrets
- All inputs validated
- RBAC enforcement

---

## Documentation

- [Foundation Docs](00-foundation/)
- [Architecture](01-architecture/)
- [SDK Guide](02-sdk/)
- [Generator](03-generator/)
- [Evaluation Framework](04-evaluation/)
- [Registry](05-registry/)

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run checks (`make check`)
5. Commit your changes (`git commit -m 'feat: add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## Support

- **Documentation**: [00-README.md](00-README.md)
- **Issues**: [GitHub Issues](https://github.com/greenlang/agent-factory/issues)
- **Discussions**: [GitHub Discussions](https://github.com/greenlang/agent-factory/discussions)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Status**: Production Ready | **Version**: 1.0.0 | **Last Updated**: December 2025
