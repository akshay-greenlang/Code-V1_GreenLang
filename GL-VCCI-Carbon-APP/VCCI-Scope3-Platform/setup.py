# GL-VCCI-Carbon-APP Setup Configuration
# Python package setup for Scope 3 Value Chain Carbon Intelligence Platform
#
# Installation:
#   pip install -e .  # Development mode (editable)
#   pip install .  # Production install
#
# Build distribution:
#   python setup.py sdist bdist_wheel
#
# Upload to PyPI (if making public):
#   twine upload dist/*

from setuptools import setup, find_packages
import os
import sys

# Ensure Python 3.9+
if sys.version_info < (3, 9):
    print("ERROR: Python 3.9+ is required")
    sys.exit(1)

# Read README for long description
def read_long_description():
    """Read README.md for package long description."""
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Scope 3 Value Chain Carbon Intelligence Platform"

# Read requirements.txt
def read_requirements():
    """Read requirements.txt and return list of dependencies."""
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    requirements = []

    if os.path.exists(req_path):
        with open(req_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith("#"):
                    requirements.append(line)

    return requirements

# Package metadata
setup(
    # Basic info
    name="vcci-scope3-platform",
    version="1.0.0",
    description="Enterprise Scope 3 Value Chain Carbon Intelligence Platform",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",

    # Author info
    author="GreenLang Framework Team",
    author_email="team@greenlang.io",
    maintainer="GL-VCCI Project Team",
    maintainer_email="vcci@greenlang.io",

    # URLs
    url="https://greenlang.io/packs/vcci-scope3",
    project_urls={
        "Documentation": "https://docs.greenlang.io/packs/vcci-scope3",
        "Source": "https://github.com/greenlang/gl-vcci-carbon-app",
        "Bug Reports": "https://github.com/greenlang/gl-vcci-carbon-app/issues",
        "Community": "https://community.greenlang.io",
    },

    # License
    license="Proprietary",

    # Classifiers (PyPI metadata)
    classifiers=[
        # Development status
        "Development Status :: 5 - Production/Stable",

        # Intended audience
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",

        # Topic
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Office/Business :: Financial :: Accounting",

        # License
        "License :: Other/Proprietary License",

        # Python versions
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",

        # Operating systems
        "Operating System :: OS Independent",

        # Framework
        "Framework :: FastAPI",

        # Environment
        "Environment :: Console",
        "Environment :: Web Environment",
    ],

    # Python version requirement
    python_requires=">=3.9",

    # Keywords
    keywords=[
        "scope3",
        "carbon-accounting",
        "ghg-protocol",
        "emissions-tracking",
        "climate",
        "esg",
        "sustainability",
        "supply-chain",
        "cdp",
        "sbti",
        "greenlang",
    ],

    # Package discovery
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),

    # Include non-Python files
    include_package_data=True,
    package_data={
        "": [
            "*.yaml",
            "*.yml",
            "*.json",
            "*.md",
            "*.txt",
            "*.html",
            "*.css",
            "*.js",
            "data/*.json",
            "data/*.yaml",
            "schemas/*.json",
            "rules/*.yaml",
            "templates/*",
        ],
    },

    # Dependencies
    install_requires=read_requirements(),

    # Optional dependencies (extras)
    extras_require={
        # Development dependencies
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "faker>=22.0.0",
            "black>=23.12.0",
            "ruff>=0.1.9",
            "mypy>=1.8.0",
            "isort>=5.13.0",
            "pre-commit>=3.6.0",
        ],

        # Documentation
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.5.0",
            "mkdocstrings[python]>=0.24.0",
        ],

        # All extras
        "all": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "faker>=22.0.0",
            "black>=23.12.0",
            "ruff>=0.1.9",
            "mypy>=1.8.0",
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.5.0",
        ],
    },

    # Entry points (CLI commands)
    entry_points={
        "console_scripts": [
            "vcci=cli.vcci_commands:main",  # Main CLI command
            "scope3=cli.vcci_commands:main",  # Alias
        ],
    },

    # Zip safety
    zip_safe=False,

    # Additional metadata
    platforms=["any"],

    # Test suite
    test_suite="tests",

    # Tests require
    tests_require=[
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0",
    ],
)
