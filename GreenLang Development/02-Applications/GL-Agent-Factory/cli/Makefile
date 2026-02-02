# Makefile for GreenLang Agent Factory CLI

.PHONY: help install dev-install test lint format clean build publish

# Default target
help:
	@echo "GreenLang Agent Factory CLI - Available targets:"
	@echo ""
	@echo "  install      - Install CLI package"
	@echo "  dev-install  - Install with development dependencies"
	@echo "  test         - Run tests with pytest"
	@echo "  lint         - Run linters (ruff, mypy)"
	@echo "  format       - Format code with black"
	@echo "  clean        - Remove build artifacts"
	@echo "  build        - Build distribution packages"
	@echo "  publish      - Publish to PyPI"
	@echo "  run          - Run CLI (gl --help)"
	@echo ""

# Installation
install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

# Testing
test:
	pytest tests/ -v --cov=cli --cov-report=term-missing --cov-report=html

test-verbose:
	pytest tests/ -vv --cov=cli --cov-report=term-missing

coverage:
	pytest tests/ --cov=cli --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

# Linting and formatting
lint:
	ruff check cli/
	mypy cli/

lint-fix:
	ruff check --fix cli/

format:
	black cli/
	ruff check --fix cli/

format-check:
	black --check cli/

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Building
build: clean
	python -m build

build-wheel:
	python -m build --wheel

build-sdist:
	python -m build --sdist

# Publishing
publish-test: build
	python -m twine upload --repository testpypi dist/*

publish: build
	python -m twine upload dist/*

# Development
run:
	gl --help

run-version:
	gl --version

run-init:
	gl init

# Type checking
typecheck:
	mypy cli/ --strict

# Security check
security:
	pip install bandit safety
	bandit -r cli/
	safety check

# Documentation
docs:
	@echo "Generating documentation..."
	# Add documentation generation commands here

# All checks (CI/CD)
check-all: format-check lint test
	@echo "All checks passed!"

# Setup development environment
setup-dev:
	python -m venv venv
	@echo "Virtual environment created."
	@echo "Activate with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"
	@echo "Then run: make dev-install"

# Update dependencies
update-deps:
	pip install --upgrade pip setuptools wheel
	pip install --upgrade -r requirements.txt

# Create release
release: check-all build
	@echo "Release package created in dist/"
	@echo "Ready to publish!"
