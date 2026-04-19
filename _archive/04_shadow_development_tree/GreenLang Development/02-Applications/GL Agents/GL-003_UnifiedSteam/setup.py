#!/usr/bin/env python3
"""
Setup script for GL-003 UNIFIEDSTEAM

This setup.py is provided for backward compatibility.
Modern builds should use pyproject.toml with pip or build.

Usage:
    pip install -e .           # Development install
    pip install -e .[dev]      # With dev dependencies
    pip install -e .[all]      # All optional dependencies
    python setup.py sdist      # Source distribution
    python setup.py bdist_wheel # Wheel distribution
"""

import os
import sys
from pathlib import Path

from setuptools import find_packages, setup

# Ensure we're in the right directory
HERE = Path(__file__).parent.resolve()
os.chdir(HERE)

# Read version from package
def get_version():
    """Extract version from pyproject.toml."""
    import re
    pyproject = HERE / "pyproject.toml"
    if pyproject.exists():
        content = pyproject.read_text()
        match = re.search(r'version\s*=\s*"([^"]+)"', content)
        if match:
            return match.group(1)
    return "1.0.0"

# Read long description
def get_long_description():
    """Read README for long description."""
    readme = HERE / "README.md"
    if readme.exists():
        return readme.read_text(encoding="utf-8")
    return ""

# Core dependencies
INSTALL_REQUIRES = [
    "pydantic>=2.0",
    "numpy>=1.24",
    "scipy>=1.10",
    "pandas>=2.0",
    "asyncio-throttle>=1.0",
    "aiohttp>=3.8",
    "aiokafka>=0.8",
    "grpcio>=1.54",
    "protobuf>=4.0",
    "fastapi>=0.100",
    "uvicorn>=0.22",
    "sqlalchemy>=2.0",
    "structlog>=23.1",
    "prometheus-client>=0.17",
]

# Optional dependencies
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.0",
        "pytest-asyncio>=0.21",
        "pytest-cov>=4.0",
        "pytest-benchmark>=4.0",
        "hypothesis>=6.75",
        "mypy>=1.3",
        "ruff>=0.0.270",
        "black>=23.3",
    ],
    "ml": [
        "scikit-learn>=1.3",
        "xgboost>=1.7",
        "shap>=0.42",
        "networkx>=3.0",
    ],
    "grpc": [
        "grpcio>=1.54",
        "grpcio-tools>=1.54",
    ],
    "kafka": [
        "aiokafka>=0.8",
        "confluent-kafka>=2.0",
    ],
    "climate": [
        "reportlab>=4.0",
        "openpyxl>=3.1",
    ],
}
EXTRAS_REQUIRE["all"] = sum(EXTRAS_REQUIRE.values(), [])

setup(
    name="gl003-unifiedsteam",
    version=get_version(),
    description="GL-003 UNIFIEDSTEAM: Unified Steam System Optimization Agent",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="GreenLang Team",
    author_email="team@greenlang.io",
    url="https://github.com/greenlang/gl003-unifiedsteam",
    project_urls={
        "Documentation": "https://docs.greenlang.io/gl003",
        "Source": "https://github.com/greenlang/gl003-unifiedsteam",
        "Issues": "https://github.com/greenlang/gl003-unifiedsteam/issues",
    },
    license="Proprietary",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Typing :: Typed",
    ],
    packages=find_packages(
        where=".",
        include=["GL_Agents*"],
        exclude=["tests*", "docs*", "examples*"],
    ),
    package_dir={"": "."},
    package_data={
        "": ["py.typed", "*.pyi", "*.json", "*.yaml", "*.proto"],
    },
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "gl003=GL_Agents.GL003_UnifiedSteam.cli:main",
            "gl003-server=GL_Agents.GL003_UnifiedSteam.server:main",
        ],
    },
    zip_safe=False,
)
