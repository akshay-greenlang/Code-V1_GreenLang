"""
Setup script for GreenLang Agent Factory CLI

This script provides backward compatibility for older pip versions
that don't support pyproject.toml.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="greenlang-agent-factory-cli",
    version="0.1.0",
    description="CLI for the GreenLang Agent Factory - Generate, validate, test, and publish AI agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="GreenLang Team",
    author_email="team@greenlang.io",
    url="https://github.com/greenlang/agent-factory",
    packages=find_packages(),
    package_data={
        "cli": ["templates/*"],
    },
    python_requires=">=3.11",
    install_requires=[
        "typer[all]>=0.12.0",
        "rich>=13.7.0",
        "pyyaml>=6.0.1",
        "pydantic>=2.5.0",
        "requests>=2.31.0",
        "jinja2>=3.1.2",
        "click>=8.1.7",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "ruff>=0.1.8",
            "mypy>=1.7.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "gl=cli.main:cli_main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
    ],
    keywords="ai agents cli greenlang factory",
)
