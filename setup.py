"""
GreenLang - The Climate Intelligence Framework
Build climate apps fast with modular agents, YAML pipelines, and Python SDK
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read version from VERSION file
version = Path("VERSION").read_text().strip()

# Read README for long description
this_directory = Path(__file__).parent
long_description_path = this_directory / "README.md"
if long_description_path.exists():
    long_description = long_description_path.read_text(encoding="utf-8")
else:
    long_description = "GreenLang - The Climate Intelligence Framework for the Entire Climate Industry"

# Core production dependencies
install_requires = [
    # Core Framework
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
    "click>=8.0.0",
    "rich>=13.0.0",
    "python-dotenv>=1.0.0",
    "typing-extensions>=4.9.0",  # Type hints support
    
    # Data Processing
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "openpyxl>=3.1.0",
    "jsonschema>=4.19.0",
    
    # API & Integration
    "httpx>=0.24.0",
    "requests>=2.31.0",
    "aiohttp>=3.8.0",
    
    # Workflow Orchestration
    "networkx>=3.0",
]

# Optional extras
extras_require = {
    # Development environment
    "dev": [
        # Testing
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-asyncio>=0.21.0",
        "pytest-mock>=3.11.0",
        "pytest-timeout>=2.1.0",
        "pytest-xdist>=3.3.0",
        "pytest-benchmark>=4.0.0",
        "hypothesis>=6.80.0",
        
        # Code Quality
        "mypy>=1.5.0",
        "ruff>=0.1.0",
        "black>=23.7.0",
        "isort>=5.12.0",
        "bandit>=1.7.0",
        "coverage[toml]>=7.2.0",
        
        # Test Utilities
        "faker>=19.0.0",
        "responses>=0.23.0",
        "freezegun>=1.2.0",
        "factory-boy>=3.3.0",
        
        # Development Tools
        "pre-commit>=3.0.0",
        "ipython>=8.0.0",
        "jupyter>=1.0.0",
        "watchdog>=3.0.0",
    ],
    
    # AI/LLM capabilities
    "ai": [
        "openai>=1.0.0",
        "langchain>=0.1.0",
        "langchain-openai>=0.0.5",
        "anthropic>=0.7.0",
    ],
    
    # API server
    "api": [
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "gunicorn>=21.0.0",
    ],
    
    # Database support
    "db": [
        "sqlalchemy>=2.0.0",
        "alembic>=1.12.0",
        "psycopg2-binary>=2.9.0",
        "pymongo>=4.5.0",
        "redis>=5.0.0",
    ],
    
    # Monitoring & observability
    "monitoring": [
        "prometheus-client>=0.18.0",
        "opentelemetry-api>=1.20.0",
        "structlog>=23.0.0",
    ],
    
    # Documentation
    "docs": [
        "mkdocs>=1.5.0",
        "mkdocs-material>=9.0.0",
        "mkdocstrings>=0.23.0",
    ],
    
    # Testing only
    "test": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-asyncio>=0.21.0",
        "pytest-mock>=3.11.0",
        "pytest-timeout>=2.1.0",
        "pytest-xdist>=3.3.0",
        "pytest-benchmark>=4.0.0",
        "hypothesis>=6.80.0",
        "faker>=19.0.0",
        "responses>=0.23.0",
        "freezegun>=1.2.0",
        "factory-boy>=3.3.0",
        # Type checking
        "mypy>=1.7.0",
        "types-PyYAML>=6.0.12",
        "types-requests>=2.31.0",
        "types-click>=8.1.0",
        "types-jsonschema>=4.19.0",
    ],
}

# All extras combined
all_extras = []
for extra_deps in extras_require.values():
    all_extras.extend(extra_deps)
extras_require["all"] = list(set(all_extras))

setup(
    name="greenlang",
    version=version,
    author="GreenLang Team",
    author_email="team@greenlang.io",
    description="Infrastructure for Climate Intelligence - Domain logic lives in packs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/greenlang/greenlang",
    project_urls={
        "Documentation": "https://docs.greenlang.ai",
        "Source": "https://github.com/greenlang/greenlang",
        "Issues": "https://github.com/greenlang/greenlang/issues",
        "Discord": "https://discord.gg/greenlang",
        "Twitter": "https://twitter.com/GreenLangAI",
    },
    
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples"]),
    
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Natural Language :: English",
    ],

    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
    
    entry_points={
        "console_scripts": [
            "greenlang=greenlang.cli.main:cli",  # Legacy CLI
            "gl=core.greenlang.cli.main:app",    # New v0.1 CLI
        ],
        "greenlang.agents": [
            "fuel=greenlang.agents.fuel_agent:FuelAgent",
            "carbon=greenlang.agents.carbon_agent:CarbonAgent",
            "intensity=greenlang.agents.intensity_agent:IntensityAgent",
            "benchmark=greenlang.agents.benchmark_agent:BenchmarkAgent",
            "grid_factor=greenlang.agents.grid_factor_agent:GridFactorAgent",
            "building_profile=greenlang.agents.building_profile_agent:BuildingProfileAgent",
            "recommendation=greenlang.agents.recommendation_agent:RecommendationAgent",
            "report=greenlang.agents.report_agent:ReportAgent",
            "validator=greenlang.agents.input_validator_agent:InputValidatorAgent",
        ],
        "greenlang.packs": [
            # Packs register themselves here when installed
            # Example: "emissions-core=packs.emissions_core:MANIFEST"
        ],
    },
    
    include_package_data=True,
    package_data={
        "greenlang": [
            "data/*.json",
            "data/**/*.json",
            "workflows/*.yaml",
            "workflows/**/*.yaml",
            "templates/*.html",
            "templates/*.md",
            "configs/*.yaml",
        ],
    },
    
    zip_safe=False,
    
    keywords=[
        "climate", "emissions", "carbon", "sustainability", "ESG",
        "greenhouse gas", "GHG", "carbon footprint", "net zero",
        "climate intelligence", "building emissions", "HVAC", "solar thermal",
        "boiler systems", "carbon accounting", "climate framework",
        "environmental", "climate tech", "green tech", "scope 1 2 3",
        "decarbonization", "climate apps", "emissions modeling"
    ],
    
    test_suite="tests",
    tests_require=[
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
    ],
)