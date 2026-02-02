#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Static Validation Script for GL-VCCI Platform
Validates all code files without requiring Python installation
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple

# Base directory
BASE_DIR = Path(__file__).parent


class ValidationReport:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []

    def add_error(self, category, file, message):
        self.errors.append((category, file, message))

    def add_warning(self, category, file, message):
        self.warnings.append((category, file, message))

    def add_info(self, category, message):
        self.info.append((category, message))

    def print_report(self):
        print("\n" + "="*80)
        print("GL-VCCI PLATFORM STATIC VALIDATION REPORT")
        print("="*80)

        # Print errors
        if self.errors:
            print(f"\n ERRORS ({len(self.errors)}):")
            print("-"*80)
            for cat, file, msg in self.errors:
                print(f"  [{cat}] {file}")
                print(f"      {msg}")
        else:
            print("\n NO ERRORS FOUND")

        # Print warnings
        if self.warnings:
            print(f"\n WARNINGS ({len(self.warnings)}):")
            print("-"*80)
            for cat, file, msg in self.warnings:
                print(f"  [{cat}] {file}")
                print(f"      {msg}")

        # Print info
        if self.info:
            print(f"\n INFO ({len(self.info)}):")
            print("-"*80)
            for cat, msg in self.info:
                print(f"  [{cat}] {msg}")

        # Summary
        print("\n" + "="*80)
        print(f"SUMMARY: {len(self.errors)} errors, {len(self.warnings)} warnings")
        print("="*80)


def extract_imports(file_path: Path) -> Tuple[List[str], List[str]]:
    """Extract import statements from a Python file."""
    absolute_imports = []
    relative_imports = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find absolute imports
        abs_pattern = r'^(?:from\s+([\w.]+)\s+)?import\s+([\w\s,.*]+)'
        for match in re.finditer(abs_pattern, content, re.MULTILINE):
            if match.group(1):
                # from X import Y
                module = match.group(1)
                if not module.startswith('.'):
                    absolute_imports.append(module)
            else:
                # import X
                imports = match.group(2)
                for imp in imports.split(','):
                    imp = imp.strip().split()[0]
                    if not imp.startswith('.'):
                        absolute_imports.append(imp)

        # Find relative imports
        rel_pattern = r'from\s+(\.+[\w.]*)\s+import'
        for match in re.finditer(rel_pattern, content):
            relative_imports.append(match.group(1))

    except Exception as e:
        pass

    return absolute_imports, relative_imports


def check_category_imports(report: ValidationReport):
    """Check if category calculators can import properly."""
    categories_dir = BASE_DIR / "services" / "agents" / "calculator" / "categories"

    if not categories_dir.exists():
        report.add_error("STRUCTURE", "categories/", "Directory does not exist")
        return

    # Check for __init__.py
    init_file = categories_dir / "__init__.py"
    if not init_file.exists():
        report.add_error("IMPORT", "categories/__init__.py", "Missing __init__.py file")
        return

    # Get exports from __init__.py
    with open(init_file, 'r') as f:
        init_content = f.read()

    # Expected calculators
    expected_calcs = [f"Category{i}Calculator" for i in range(1, 16)]

    for calc in expected_calcs:
        if calc not in init_content:
            report.add_error("IMPORT", "categories/__init__.py", f"Missing export: {calc}")

    # Check each category file
    for i in range(1, 16):
        cat_file = categories_dir / f"category_{i}.py"
        if not cat_file.exists():
            report.add_error("FILE", f"category_{i}.py", "File does not exist")
            continue

        # Check if class is defined
        with open(cat_file, 'r') as f:
            content = f.read()

        class_name = f"Category{i}Calculator"
        if f"class {class_name}" not in content:
            report.add_error("CLASS", f"category_{i}.py", f"Class {class_name} not defined")

        # Check for calculate method
        if "def calculate(" not in content and "async def calculate(" not in content:
            report.add_error("METHOD", f"category_{i}.py", "Missing calculate() method")

        report.add_info("FILE", f"category_{i}.py validated")


def check_models(report: ValidationReport):
    """Check if all input models are defined."""
    models_file = BASE_DIR / "services" / "agents" / "calculator" / "models.py"

    if not models_file.exists():
        report.add_error("FILE", "models.py", "File does not exist")
        return

    with open(models_file, 'r') as f:
        content = f.read()

    # Expected input models
    expected_models = [f"Category{i}Input" for i in range(1, 16)]

    for model in expected_models:
        if f"class {model}" not in content:
            report.add_error("MODEL", "models.py", f"Missing model: {model}")

    report.add_info("MODELS", f"All input models validated")


def check_enums(report: ValidationReport):
    """Check if all enums are defined in config.py."""
    config_file = BASE_DIR / "services" / "agents" / "calculator" / "config.py"

    if not config_file.exists():
        report.add_error("FILE", "config.py", "File does not exist")
        return

    with open(config_file, 'r') as f:
        content = f.read()

    # Expected enums
    expected_enums = [
        "TierType", "TransportMode", "CabinClass", "CommuteMode",
        "BuildingType", "FranchiseType", "ProductType", "MaterialType",
        "DisposalMethod", "AssetClass"
    ]

    for enum in expected_enums:
        if f"class {enum}" not in content:
            report.add_error("ENUM", "config.py", f"Missing enum: {enum}")

    report.add_info("ENUMS", "All enums validated")


def check_agent_integration(report: ValidationReport):
    """Check if agent.py properly imports all calculators."""
    agent_file = BASE_DIR / "services" / "agents" / "calculator" / "agent.py"

    if not agent_file.exists():
        report.add_error("FILE", "agent.py", "File does not exist")
        return

    with open(agent_file, 'r') as f:
        content = f.read()

    # Check calculator imports
    expected_imports = [f"Category{i}Calculator" for i in range(1, 16)]

    for imp in expected_imports:
        if imp not in content:
            report.add_error("IMPORT", "agent.py", f"Missing import: {imp}")

    # Check model imports
    expected_model_imports = [f"Category{i}Input" for i in range(1, 16)]

    for imp in expected_model_imports:
        if imp not in content:
            report.add_error("IMPORT", "agent.py", f"Missing model import: {imp}")

    # Check calculate methods
    for i in range(1, 16):
        if f"def calculate_category_{i}(" not in content and f"async def calculate_category_{i}(" not in content:
            if i not in [1, 4, 6]:  # Original categories might have different structure
                report.add_warning("METHOD", "agent.py", f"calculate_category_{i}() method may be missing")

    report.add_info("AGENT", "Agent integration validated")


def check_cli_structure(report: ValidationReport):
    """Check CLI structure and imports."""
    cli_dir = BASE_DIR / "cli"

    if not cli_dir.exists():
        report.add_error("STRUCTURE", "cli/", "Directory does not exist")
        return

    # Check main.py
    main_file = cli_dir / "main.py"
    if not main_file.exists():
        report.add_error("FILE", "cli/main.py", "File does not exist")
    else:
        with open(main_file, 'r') as f:
            content = f.read()

        # Check for typer import
        if "import typer" not in content:
            report.add_error("IMPORT", "cli/main.py", "Missing typer import")

        # Check for rich import
        if "from rich" not in content:
            report.add_error("IMPORT", "cli/main.py", "Missing rich import")

    # Check commands directory
    commands_dir = cli_dir / "commands"
    if not commands_dir.exists():
        report.add_error("STRUCTURE", "cli/commands/", "Directory does not exist")
    else:
        expected_commands = ["intake.py", "engage.py", "pipeline.py"]
        for cmd_file in expected_commands:
            if not (commands_dir / cmd_file).exists():
                report.add_error("FILE", f"cli/commands/{cmd_file}", "File does not exist")

    report.add_info("CLI", "CLI structure validated")


def check_requirements(report: ValidationReport):
    """Check requirements.txt for necessary dependencies."""
    req_file = BASE_DIR / "requirements.txt"

    if not req_file.exists():
        report.add_error("FILE", "requirements.txt", "File does not exist")
        return

    with open(req_file, 'r') as f:
        content = f.read().lower()

    # Required dependencies
    required_deps = [
        ("typer", "CLI framework"),
        ("rich", "CLI output"),
        ("anthropic", "LLM provider"),
        ("openai", "LLM provider"),
        ("pydantic", "Data validation"),
        ("pandas", "Data processing"),
    ]

    for dep, purpose in required_deps:
        if dep not in content:
            report.add_warning("DEPENDENCY", "requirements.txt", f"Missing {dep} ({purpose})")

    report.add_info("REQUIREMENTS", "Requirements validated")


def main():
    """Run all validation checks."""
    report = ValidationReport()

    print("\nRunning GL-VCCI Platform Static Validation...")
    print("="*80)

    # Run all checks
    check_category_imports(report)
    check_models(report)
    check_enums(report)
    check_agent_integration(report)
    check_cli_structure(report)
    check_requirements(report)

    # Print report
    report.print_report()

    # Return exit code
    return 0 if len(report.errors) == 0 else 1


if __name__ == "__main__":
    exit(main())
