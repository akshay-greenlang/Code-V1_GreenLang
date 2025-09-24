#!/usr/bin/env python3
"""
Setup script for GreenLang CLI

This file is kept for legacy compatibility and pip editable installs.
The main package configuration is in pyproject.toml.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read version from VERSION file
version_file = Path(__file__).parent / "VERSION"
if version_file.exists():
    version = version_file.read_text().strip()
else:
    version = "0.3.0"

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")
else:
    long_description = "GreenLang Climate Intelligence Platform"

setup(
    name="greenlang-cli",
    version=version,
    description="Climate Intelligence Platform - Enterprise infrastructure with powerful SDK for climate-aware applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="GreenLang Maintainers",
    author_email="maintainers@greenlang.io",
    url="https://greenlang.io",
    project_urls={
        "Homepage": "https://greenlang.io",
        "Documentation": "https://greenlang.io/docs",
        "Repository": "https://github.com/greenlang/greenlang",
        "Bug Tracker": "https://github.com/greenlang/greenlang/issues",
        "Changelog": "https://github.com/greenlang/greenlang/blob/main/CHANGELOG.md",
        "Discord": "https://discord.gg/greenlang",
    },
    license="MIT",
    packages=find_packages(where=".", include=["greenlang*", "core*"]),
    package_data={
        "greenlang": [
            "schemas/*.json",
            "data/**/*.json",
            "examples/**/*",
        ],
        "core.greenlang": [
            "schemas/*.json",
            "data/**/*.json",
            "configs/*.yaml",
            "policy/bundles/**/*",
        ],
    },
    python_requires=">=3.10",
    install_requires=[
        "typer>=0.12",
        "pydantic>=2.7",
        "pyyaml>=6.0",
        "rich>=13.0",
        "jsonschema>=4.19.0",
        "packaging>=22.0",
        "python-dotenv>=1.0.0",
        "typing-extensions>=4.9.0",
        "httpx>=0.24.0",
        "requests>=2.31.0",
        "networkx>=3.0",
        "tenacity>=8.2.3",
        "psutil>=5.9.0",
    ],
    extras_require={
        "analytics": [
            "pandas>=2.0.0",
            "numpy>=1.24.0",
        ],
        "cli": [
            "click>=8.0.0",
            "rich>=13.0.0",
        ],
        "data": [
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "openpyxl>=3.1.0",
            "jinja2>=3.1.0",
            "weasyprint>=60.0",
            "sqlalchemy>=2.0.0",
            "alembic>=1.12.0",
            "psycopg2-binary>=2.9.0",
            "pymongo>=4.5.0",
        ],
        "llm": [
            "openai>=1.0.0",
            "langchain>=0.1.0",
            "langchain-openai>=0.0.5",
            "langchain-community>=0.0.10",
            "anthropic>=0.7.0",
            "faiss-cpu>=1.7.4",
            "sentence-transformers>=2.2.0",
            "pypdf>=3.17.0",
            "chromadb>=0.4.0",
        ],
        "server": [
            "aiohttp>=3.8.0",
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "redis>=5.0.0",
            "celery>=5.3.0",
            "prometheus-client>=0.18.0",
            "opentelemetry-api>=1.20.0",
            "structlog>=23.0.0",
            "gunicorn>=21.0.0",
        ],
        "security": [
            "cryptography>=41.0.0",
            "PyJWT>=2.8.0",
        ],
        "test": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.11.0",
            "pytest-timeout>=2.1.0",
            "pytest-xdist>=3.3.0",
            "pytest-benchmark>=4.0.0",
            "pytest-regressions>=2.5.0",
            "hypothesis>=6.80.0",
            "psutil>=5.9.0",
            "coverage[toml]>=7.2.0",
            "faker>=19.0.0",
            "responses>=0.23.0",
            "freezegun>=1.2.0",
            "factory-boy>=3.3.0",
        ],
        "dev": [
            "mypy>=1.7.0",
            "ruff>=0.1.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "bandit>=1.7.0",
            "types-PyYAML>=6.0.12",
            "types-requests>=2.31.0",
            "types-redis>=4.6.0",
            "types-jsonschema>=4.19.0",
            "ipython>=8.0.0",
            "jupyter>=1.0.0",
            "pre-commit>=3.0.0",
            "watchdog>=3.0.0",
            "docker>=6.1.0",
            "kubernetes>=28.0.0",
            "flake8>=6.0",
        ],
        "doc": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.0.0",
            "mkdocstrings>=0.23.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gl=greenlang.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Framework :: Pydantic",
        "Natural Language :: English",
    ],
    keywords=[
        "climate",
        "emissions",
        "sustainability",
        "carbon",
        "green",
        "environment",
        "AI",
        "orchestration",
        "platform",
        "infrastructure",
        "framework",
        "SDK",
        "climate-intelligence",
        "decarbonization",
        "net-zero",
        "ESG",
        "HVAC",
        "buildings",
        "energy",
    ],
    zip_safe=False,
)