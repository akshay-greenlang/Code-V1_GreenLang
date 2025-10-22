"""
Setup script for CSRD/ESRS Digital Reporting Platform.

Installation:
    pip install -e .

Development installation:
    pip install -e ".[dev]"
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="csrd-reporting-platform",
    version="1.0.0",
    description="CSRD/ESRS Digital Reporting Platform - Zero-hallucination EU sustainability reporting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="GreenLang CSRD Team",
    author_email="csrd@greenlang.io",
    url="https://github.com/akshay-greenlang/Code-V1_GreenLang/tree/master/GL-CSRD-APP",
    license="MIT",

    # Package configuration
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    include_package_data=True,
    package_data={
        "": [
            "data/*.json",
            "data/*.yaml",
            "data/*.csv",
            "schemas/*.json",
            "rules/*.yaml",
            "config/*.yaml",
        ],
    },

    # Dependencies
    install_requires=requirements,

    # Development dependencies
    extras_require={
        "dev": [
            "pytest>=7.4.4",
            "pytest-cov>=4.1.0",
            "black>=24.1.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
            "ipython>=8.20.0",
            "jupyter>=1.0.0",
        ],
        "ai": [
            "openai>=1.10.0",
            "anthropic>=0.18.0",
            "pinecone-client>=3.0.0",
        ],
        "full": [
            "openai>=1.10.0",
            "anthropic>=0.18.0",
            "pinecone-client>=3.0.0",
            "pytest>=7.4.4",
            "pytest-cov>=4.1.0",
        ],
    },

    # Python version requirement
    python_requires=">=3.11",

    # Entry points for CLI
    entry_points={
        "console_scripts": [
            "csrd=cli.csrd_commands:csrd",
        ],
    },

    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Legal Industry",
        "Topic :: Office/Business :: Financial",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Typing :: Typed",
    ],

    # Keywords
    keywords=[
        "csrd",
        "esrs",
        "sustainability",
        "esg",
        "reporting",
        "xbrl",
        "compliance",
        "eu-regulation",
        "zero-hallucination",
        "greenlang",
    ],

    # Project URLs
    project_urls={
        "Documentation": "https://github.com/akshay-greenlang/Code-V1_GreenLang/tree/master/GL-CSRD-APP/docs",
        "Source": "https://github.com/akshay-greenlang/Code-V1_GreenLang/tree/master/GL-CSRD-APP",
        "Bug Reports": "https://github.com/akshay-greenlang/Code-V1_GreenLang/issues",
    },
)
