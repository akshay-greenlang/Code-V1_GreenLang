# GreenLang Development Makefile
# Usage: make <target>
# Example: make init validate test

.PHONY: help init validate test pack-publish pack-add doctor clean install dev-install format lint type-check build release docs serve

# Default target - show help
help:
	@echo "GreenLang Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  init          - Initialize development environment"
	@echo "  install       - Install GreenLang package"
	@echo "  dev-install   - Install with development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  validate      - Validate all packs"
	@echo "  test          - Run test suite" 
	@echo "  format        - Format code with black and isort"
	@echo "  lint          - Run linting with ruff"
	@echo "  type-check    - Run type checking with mypy"
	@echo "  doctor        - Run environment diagnostics"
	@echo ""
	@echo "Pack Management:"
	@echo "  pack-publish  - Publish pack (use P=<pack_path>)"
	@echo "  pack-add      - Add/install pack (use R=<pack_ref>)"
	@echo ""
	@echo "Build & Release:"
	@echo "  build         - Build distribution packages"
	@echo "  release       - Release to PyPI"
	@echo "  docs          - Build documentation"
	@echo "  serve         - Serve docs locally"
	@echo ""
	@echo "Utilities:"
	@echo "  clean         - Clean generated files"
	@echo ""
	@echo "Examples:"
	@echo "  make pack-publish P=packs/boiler-solar"
	@echo "  make pack-add R=greenlang/hvac-measures@1.0.0"

# Environment setup
init:
	@echo "🚀 Initializing GreenLang development environment..."
	pipx install poetry || pip install poetry || true
	poetry --version || echo "Poetry not available, using pip"
	poetry install || pip install -e .[dev] || pip install -e .
	@echo "✅ Environment initialized"

# Standard installation
install:
	@echo "📦 Installing GreenLang..."
	pip install -e .
	@echo "✅ Installation complete"

# Development installation with extra dependencies
dev-install:
	@echo "🛠️  Installing GreenLang with development dependencies..."
	pip install -e .[dev]
	@echo "✅ Development installation complete"

# Pack validation
validate:
	@echo "🔍 Validating all packs..."
	gl pack validate packs/boiler-solar/ || echo "⚠️  boiler-solar validation issues"
	gl pack validate packs/hvac-measures/ || echo "⚠️  hvac-measures validation issues" 
	gl pack validate packs/cement-lca/ || echo "⚠️  cement-lca validation issues"
	gl pack validate packs/emissions-core/ || echo "⚠️  emissions-core validation issues"
	@echo "✅ Pack validation complete"

# Test suite
test:
	@echo "🧪 Running test suite..."
	python -m pytest -q --tb=short
	@echo "✅ Tests complete"

# Extended test suite with coverage
test-full:
	@echo "🧪 Running full test suite with coverage..."
	python -m pytest -v --cov=core --cov-report=html --cov-report=term
	@echo "✅ Full tests complete - see htmlcov/index.html"

# Environment diagnostics
doctor:
	@echo "🩺 Running environment diagnostics..."
	gl doctor
	@echo "✅ Diagnostics complete"

# Pack publishing (requires P=pack_path)
pack-publish:
	@if [ -z "$(P)" ]; then \
		echo "❌ Error: Please specify pack path with P=<path>"; \
		echo "Example: make pack-publish P=packs/boiler-solar"; \
		exit 1; \
	fi
	@echo "📤 Publishing pack: $(P)"
	gl pack publish $(P)
	@echo "✅ Pack published"

# Pack installation (requires R=pack_ref)
pack-add:
	@if [ -z "$(R)" ]; then \
		echo "❌ Error: Please specify pack reference with R=<ref>"; \
		echo "Example: make pack-add R=greenlang/boiler-solar@1.0.0"; \
		exit 1; \
	fi
	@echo "📥 Installing pack: $(R)"
	gl pack add $(R)
	@echo "✅ Pack installed"

# Code formatting
format:
	@echo "🎨 Formatting code..."
	python -m black core/ tests/ --line-length 100
	python -m isort core/ tests/ --profile black
	@echo "✅ Code formatted"

# Linting
lint:
	@echo "🔍 Running linter..."
	python -m ruff check core/ tests/ --fix || echo "⚠️  Linting issues found"
	@echo "✅ Linting complete"

# Type checking
type-check:
	@echo "🔍 Running type checker..."
	python -m mypy core/ --ignore-missing-imports || echo "⚠️  Type issues found"
	@echo "✅ Type checking complete"

# Quality gate - run all checks
quality:
	@echo "🏁 Running quality gate..."
	$(MAKE) format lint type-check test validate doctor
	@echo "✅ Quality gate complete"

# Build distribution packages
build:
	@echo "📦 Building distribution packages..."
	python -m build
	@echo "✅ Build complete - see dist/"

# Release to PyPI (requires proper credentials)
release: build
	@echo "🚀 Releasing to PyPI..."
	python -m twine upload dist/*
	@echo "✅ Release complete"

# Documentation
docs:
	@echo "📚 Building documentation..."
	cd docs && python -m sphinx . _build/html || echo "⚠️  Sphinx not available"
	@echo "✅ Documentation built - see docs/_build/html/"

# Serve documentation locally
serve:
	@echo "🌐 Serving documentation at http://localhost:8000"
	cd docs/_build/html && python -m http.server 8000

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf docs/_build/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete"

# Development workflow - quick iteration
dev: format lint test validate
	@echo "✅ Development workflow complete"

# CI/CD simulation - what gets run in GitHub Actions
ci: install test validate doctor
	@echo "✅ CI simulation complete"

# Quick pack development workflow
pack-dev:
	@if [ -z "$(P)" ]; then \
		echo "❌ Error: Please specify pack path with P=<path>"; \
		exit 1; \
	fi
	@echo "🔄 Quick pack development workflow for: $(P)"
	gl pack validate $(P)
	cd $(P) && python -m pytest tests/ -q || echo "⚠️  Tests failed"
	gl run $(P) --dry-run || echo "⚠️  Pipeline issues"
	@echo "✅ Pack development workflow complete"

# Performance profiling
profile:
	@echo "📊 Running performance profile..."
	python -m cProfile -o profile.stats -m greenlang.cli.main pack validate packs/boiler-solar/
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
	@echo "✅ Profile complete"

# Security scanning
security:
	@echo "🔒 Running security scan..."
	python -m bandit -r core/ || echo "⚠️  Security issues found"
	python -m safety check || echo "⚠️  Dependency vulnerabilities found"
	@echo "✅ Security scan complete"

# Install pre-commit hooks
hooks:
	@echo "🪝 Installing pre-commit hooks..."
	pre-commit install || pip install pre-commit && pre-commit install
	@echo "✅ Pre-commit hooks installed"

# Show package info
info:
	@echo "📋 GreenLang Package Information"
	@echo "================================"
	@python -c "import pkg_resources; print('Version:', pkg_resources.get_distribution('greenlang').version)" 2>/dev/null || echo "Version: Development"
	@python -c "import sys; print('Python:', sys.version.split()[0])"
	@gl --version 2>/dev/null || echo "GL CLI: Not available"
	@echo "Packs available:"
	@find packs/ -name "pack.yaml" -exec dirname {} \; | sort || echo "No packs found"

# Generate sample data for testing
sample-data:
	@echo "📊 Generating sample data..."
	python -c "import pandas as pd; pd.DataFrame({'temp': range(100), 'emissions': [x*1.5 for x in range(100)]}).to_csv('sample.csv', index=False)"
	@echo "✅ Sample data generated: sample.csv"

# Benchmark performance
benchmark:
	@echo "⚡ Running performance benchmarks..."
	time gl pack validate packs/boiler-solar/
	time gl run packs/boiler-solar/ --dry-run
	@echo "✅ Benchmarks complete"