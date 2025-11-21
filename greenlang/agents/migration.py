"""
Agent Migration Utilities.

This module provides tools to help migrate from deprecated agent implementations
to canonical versions.

Example:
    >>> from greenlang.agents.migration import update_imports
    >>> update_imports('my_project/', dry_run=True)
"""

import os
import re
import ast
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)


class ImportMigrator:
    """Utility to migrate agent imports to canonical versions."""

    # Mapping of old imports to new imports
    IMPORT_MAPPINGS = {
        # FuelAgent family
        "from greenlang.agents.fuel_agent import": "from greenlang.agents.fuel_agent_ai_v2 import",
        "from greenlang.agents.fuel_agent_ai import": "from greenlang.agents.fuel_agent_ai_v2 import",
        "from greenlang.agents.fuel_agent_ai_sync import": "from greenlang.agents.fuel_agent_ai_v2 import",
        "from greenlang.agents.fuel_agent_ai_async import": "from greenlang.agents.fuel_agent_ai_v2 import",

        # BoilerReplacementAgent family
        "from greenlang.agents.boiler_replacement_agent_ai import": "from greenlang.agents.boiler_replacement_agent_ai_v4 import",
        "from greenlang.agents.boiler_replacement_agent_ai_v3 import": "from greenlang.agents.boiler_replacement_agent_ai_v4 import",

        # CarbonAgent family
        "from greenlang.agents.carbon_agent import": "from greenlang.agents.carbon_agent_ai import",

        # GridFactorAgent family
        "from greenlang.agents.grid_factor_agent import": "from greenlang.agents.grid_factor_agent_ai import",

        # RecommendationAgent family
        "from greenlang.agents.recommendation_agent import": "from greenlang.agents.recommendation_agent_ai_v2 import",
        "from greenlang.agents.recommendation_agent_ai import": "from greenlang.agents.recommendation_agent_ai_v2 import",

        # ReportAgent family
        "from greenlang.agents.report_agent import": "from greenlang.agents.report_agent_ai import",

        # IndustrialHeatPumpAgent family
        "from greenlang.agents.industrial_heat_pump_agent_ai import": "from greenlang.agents.industrial_heat_pump_agent_ai_v4 import",
        "from greenlang.agents.industrial_heat_pump_agent_ai_v3 import": "from greenlang.agents.industrial_heat_pump_agent_ai_v4 import",

        # WasteHeatRecoveryAgent family
        "from greenlang.agents.waste_heat_recovery_agent_ai import": "from greenlang.agents.waste_heat_recovery_agent_ai_v3 import",

        # DecarbonizationRoadmapAgent family
        "from greenlang.agents.decarbonization_roadmap_agent_ai import": "from greenlang.agents.decarbonization_roadmap_agent_ai_v3 import",

        # BenchmarkAgent family
        "from greenlang.agents.benchmark_agent import": "from greenlang.agents.benchmark_agent_ai import",
    }

    # Class name mappings
    CLASS_MAPPINGS = {
        # Async variants
        "FuelAgentAsync": "FuelAgent",
        "FuelAgentAI": "FuelAgent",
        "FuelAgentSync": "FuelAgent",

        # Version variants
        "BoilerReplacementAgentAI": "BoilerReplacementAgent",
        "BoilerReplacementAgentV3": "BoilerReplacementAgent",
        "BoilerReplacementAgentV4": "BoilerReplacementAgent",

        # AI variants
        "CarbonAgentBase": "CarbonAgent",
        "GridFactorAgentBase": "GridFactorAgent",
        "RecommendationAgentBase": "RecommendationAgent",
        "ReportAgentBase": "ReportAgent",
    }

    # Constructor parameter mappings
    PARAM_MAPPINGS = {
        "FuelAgent": {
            "additions": {"mode": "'sync'", "ai_enabled": "True"},
            "removals": ["async_mode", "use_ai"],
        },
        "BoilerReplacementAgent": {
            "additions": {"version": "4"},
            "removals": ["legacy_mode"],
        },
    }

    def __init__(self):
        """Initialize the import migrator."""
        self.changes_made = []
        self.files_processed = 0
        self.errors = []

    def migrate_file(self, filepath: str, dry_run: bool = True) -> List[str]:
        """
        Migrate imports in a single Python file.

        Args:
            filepath: Path to Python file
            dry_run: If True, don't write changes

        Returns:
            List of changes made
        """
        changes = []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                original_content = content

            # Apply import mappings
            for old_import, new_import in self.IMPORT_MAPPINGS.items():
                if old_import in content:
                    content = content.replace(old_import, new_import)
                    changes.append(f"Updated import: {old_import} -> {new_import}")

            # Apply class name mappings
            for old_class, new_class in self.CLASS_MAPPINGS.items():
                # Match class instantiation
                pattern = rf'\b{old_class}\s*\('
                if re.search(pattern, content):
                    content = re.sub(pattern, f'{new_class}(', content)
                    changes.append(f"Updated class: {old_class} -> {new_class}")

                # Match type hints
                pattern = rf':\s*{old_class}\b'
                if re.search(pattern, content):
                    content = re.sub(pattern, f': {new_class}', content)

            # Add migration warnings
            if changes and not dry_run:
                # Add warning comment at top of file
                warning = (
                    "# WARNING: This file has been automatically migrated to use new agent APIs.\n"
                    "# Please review the changes and update any custom logic as needed.\n"
                    "# See greenlang/agents/AGENT_CONSOLIDATION_GUIDE.md for details.\n\n"
                )
                if warning not in content:
                    content = warning + content

            # Write changes if not dry run
            if changes and not dry_run:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Updated {filepath} with {len(changes)} changes")
            elif changes:
                logger.info(f"Would update {filepath} with {len(changes)} changes (dry run)")

            self.changes_made.extend([(filepath, change) for change in changes])

        except Exception as e:
            error_msg = f"Error processing {filepath}: {e}"
            self.errors.append(error_msg)
            logger.error(error_msg)

        return changes

    def migrate_directory(self, directory: str, dry_run: bool = True) -> Dict[str, List[str]]:
        """
        Migrate all Python files in a directory.

        Args:
            directory: Path to directory
            dry_run: If True, don't write changes

        Returns:
            Dictionary mapping files to changes
        """
        results = {}
        directory = Path(directory)

        for filepath in directory.rglob("*.py"):
            # Skip migration files and tests
            if "migration" in str(filepath) or "__pycache__" in str(filepath):
                continue

            changes = self.migrate_file(str(filepath), dry_run)
            if changes:
                results[str(filepath)] = changes
            self.files_processed += 1

        return results

    def generate_report(self) -> str:
        """Generate migration report."""
        report = []
        report.append("=" * 60)
        report.append("Agent Import Migration Report")
        report.append("=" * 60)
        report.append(f"Files processed: {self.files_processed}")
        report.append(f"Total changes: {len(self.changes_made)}")
        report.append(f"Errors: {len(self.errors)}")
        report.append("")

        if self.changes_made:
            report.append("Changes by file:")
            report.append("-" * 40)
            current_file = None
            for filepath, change in self.changes_made:
                if filepath != current_file:
                    report.append(f"\n{filepath}:")
                    current_file = filepath
                report.append(f"  - {change}")

        if self.errors:
            report.append("\nErrors encountered:")
            report.append("-" * 40)
            for error in self.errors:
                report.append(f"  - {error}")

        return "\n".join(report)


def update_imports(directory: str, dry_run: bool = True, verbose: bool = False) -> None:
    """
    Update agent imports in a directory.

    Args:
        directory: Path to directory to process
        dry_run: If True, don't write changes
        verbose: If True, print detailed output
    """
    migrator = ImportMigrator()

    print(f"{'[DRY RUN] ' if dry_run else ''}Scanning {directory} for deprecated agent imports...")

    results = migrator.migrate_directory(directory, dry_run)

    if verbose:
        print(migrator.generate_report())
    else:
        print(f"Found {len(results)} files with deprecated imports")
        if results and dry_run:
            print("Run with dry_run=False to apply changes")


def check_deprecated(directory: str = ".") -> List[Tuple[str, int, str]]:
    """
    Check for deprecated agent imports without modifying files.

    Args:
        directory: Directory to check

    Returns:
        List of (filepath, line_number, deprecated_import) tuples
    """
    deprecated_found = []
    directory = Path(directory)

    deprecated_patterns = [
        r"from greenlang\.agents\.fuel_agent import",
        r"from greenlang\.agents\.fuel_agent_ai import",
        r"from greenlang\.agents\.fuel_agent_ai_sync import",
        r"from greenlang\.agents\.fuel_agent_ai_async import",
        r"from greenlang\.agents\.boiler_replacement_agent_ai import",
        r"from greenlang\.agents\.boiler_replacement_agent_ai_v3 import",
        r"from greenlang\.agents\.carbon_agent import",
        r"from greenlang\.agents\.grid_factor_agent import",
        r"from greenlang\.agents\.recommendation_agent import",
        r"from greenlang\.agents\.recommendation_agent_ai import",
        r"from greenlang\.agents\.report_agent import",
        r"from greenlang\.agents\.industrial_heat_pump_agent_ai import",
        r"from greenlang\.agents\.industrial_heat_pump_agent_ai_v3 import",
        r"from greenlang\.agents\.waste_heat_recovery_agent_ai import",
        r"from greenlang\.agents\.decarbonization_roadmap_agent_ai import",
        r"from greenlang\.agents\.benchmark_agent import",
    ]

    for filepath in directory.rglob("*.py"):
        if "__pycache__" in str(filepath):
            continue

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                for pattern in deprecated_patterns:
                    if re.search(pattern, line):
                        deprecated_found.append((str(filepath), line_num, line.strip()))

        except Exception as e:
            logger.warning(f"Could not check {filepath}: {e}")

    return deprecated_found


def validate() -> bool:
    """
    Validate that migration is successful.

    Returns:
        True if validation passes
    """
    from greenlang.agents.registry import AgentRegistry

    print("Validating agent migration...")

    # Check registry is accessible
    try:
        registry = AgentRegistry()
        agents = registry.list_agents()
        print(f"✓ Registry loaded with {len(agents)} active agents")
    except Exception as e:
        print(f"✗ Registry failed to load: {e}")
        return False

    # Check canonical imports work
    # SECURITY FIX: Replaced exec() with direct imports
    import_tests = [
        ("greenlang.agents.fuel_agent_ai_v2", "FuelAgent"),
        ("greenlang.agents.boiler_replacement_agent_ai_v4", "BoilerReplacementAgent"),
        ("greenlang.agents.carbon_agent_ai", "CarbonAgent"),
    ]

    for module_name, class_name in import_tests:
        try:
            import importlib
            module = importlib.import_module(module_name)
            getattr(module, class_name)
            print(f"✓ from {module_name} import {class_name}")
        except ImportError as e:
            print(f"✗ from {module_name} import {class_name}: {e}")
            return False

    # Check deprecated warnings work
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            # This should trigger deprecation warning
            info = registry.get_agent_info("FuelAgent_legacy")
            if info and info.is_deprecated:
                print("✓ Deprecation warnings configured")
            else:
                print("✗ Deprecation info missing")
                return False
        except Exception as e:
            print(f"✗ Deprecation check failed: {e}")
            return False

    print("\n✓ All validation checks passed!")
    return True


def add_deprecation_warning(filepath: str, message: str) -> None:
    """
    Add deprecation warning to a Python file.

    Args:
        filepath: Path to file
        message: Deprecation message
    """
    warning_code = f'''
import warnings
warnings.warn(
    "{message}",
    DeprecationWarning,
    stacklevel=2
)
'''

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Add after imports
        lines = content.split('\n')
        import_end = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                import_end = i + 1
            elif import_end > 0 and line and not line.startswith('#'):
                break

        lines.insert(import_end + 1, warning_code)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        logger.info(f"Added deprecation warning to {filepath}")

    except Exception as e:
        logger.error(f"Failed to add deprecation warning to {filepath}: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m greenlang.agents.migration <command> [options]")
        print("Commands:")
        print("  validate       - Validate migration setup")
        print("  check [dir]    - Check for deprecated imports")
        print("  update [dir]   - Update deprecated imports (dry run)")
        print("  update! [dir]  - Update deprecated imports (apply changes)")
        sys.exit(1)

    command = sys.argv[1]

    if command == "validate":
        success = validate()
        sys.exit(0 if success else 1)

    elif command == "check":
        directory = sys.argv[2] if len(sys.argv) > 2 else "."
        deprecated = check_deprecated(directory)
        if deprecated:
            print(f"Found {len(deprecated)} deprecated imports:")
            for filepath, line_num, import_line in deprecated:
                print(f"  {filepath}:{line_num}: {import_line}")
        else:
            print("No deprecated imports found")

    elif command == "update":
        directory = sys.argv[2] if len(sys.argv) > 2 else "."
        update_imports(directory, dry_run=True, verbose=True)

    elif command == "update!":
        directory = sys.argv[2] if len(sys.argv) > 2 else "."
        response = input("This will modify files. Continue? (yes/no): ")
        if response.lower() == "yes":
            update_imports(directory, dry_run=False, verbose=True)
        else:
            print("Cancelled")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)