#!/usr/bin/env python3
"""
Update imports to reflect the new consolidated directory structure.
"""

import os
import re
from pathlib import Path

# Mapping of old import paths to new import paths
IMPORT_MAPPINGS = {
    # Integration
    r'from greenlang\.api\.': 'from greenlang.integration.api.',
    r'from greenlang\.sdk\.': 'from greenlang.integration.sdk.',
    r'from greenlang\.connectors\.': 'from greenlang.integration.connectors.',
    r'from greenlang\.integrations\.': 'from greenlang.integration.integrations.',
    r'from greenlang\.services\.': 'from greenlang.integration.services.',
    r'from greenlang\.adapters\.': 'from greenlang.integration.adapters.',

    # Execution
    r'from greenlang\.core\.': 'from greenlang.execution.core.',
    r'from greenlang\.pipeline\.': 'from greenlang.execution.pipeline.',
    r'from greenlang\.runtime\.': 'from greenlang.execution.runtime.',
    r'from greenlang\.infrastructure\.': 'from greenlang.execution.infrastructure.',
    r'from greenlang\.resilience\.': 'from greenlang.execution.resilience.',

    # Governance
    r'from greenlang\.policy\.': 'from greenlang.governance.policy.',
    r'from greenlang\.safety\.': 'from greenlang.governance.safety.',
    r'from greenlang\.compliance\.': 'from greenlang.governance.compliance.',
    r'from greenlang\.security\.': 'from greenlang.governance.security.',
    r'from greenlang\.validation\.': 'from greenlang.governance.validation.',

    # Utilities
    r'from greenlang\.utils\.': 'from greenlang.utilities.utils.',
    r'from greenlang\.io\.': 'from greenlang.utilities.io.',
    r'from greenlang\.serialization\.': 'from greenlang.utilities.serialization.',
    r'from greenlang\.determinism\.': 'from greenlang.utilities.determinism.',
    r'from greenlang\.lineage\.': 'from greenlang.utilities.lineage.',
    r'from greenlang\.provenance\.': 'from greenlang.utilities.provenance.',
    r'from greenlang\.cache\.': 'from greenlang.utilities.cache.',
    r'from greenlang\.tools\.': 'from greenlang.utilities.tools.',
    r'from greenlang\.visualization\.': 'from greenlang.utilities.visualization.',
    r'from greenlang\.cards\.': 'from greenlang.utilities.cards.',
    r'from greenlang\.factory\.': 'from greenlang.utilities.factory.',
    r'from greenlang\.generator\.': 'from greenlang.utilities.generator.',
    r'from greenlang\.compat\.': 'from greenlang.utilities.compat.',
    r'from greenlang\.i18n\.': 'from greenlang.utilities.i18n.',
    r'from greenlang\.exceptions\.': 'from greenlang.utilities.exceptions.',

    # Ecosystem
    r'from greenlang\.marketplace\.': 'from greenlang.ecosystem.marketplace.',
    r'from greenlang\.packs\.': 'from greenlang.ecosystem.packs.',
    r'from greenlang\.hub\.': 'from greenlang.ecosystem.hub.',
    r'from greenlang\.partners\.': 'from greenlang.ecosystem.partners.',
    r'from greenlang\.whitelabel\.': 'from greenlang.ecosystem.whitelabel.',

    # Extensions
    r'from greenlang\.ml\.': 'from greenlang.extensions.ml.',
    r'from greenlang\.ml_platform\.': 'from greenlang.extensions.ml_platform.',
    r'from greenlang\.llm\.': 'from greenlang.extensions.llm.',
    r'from greenlang\.simulation\.': 'from greenlang.extensions.simulation.',
    r'from greenlang\.business\.': 'from greenlang.extensions.business.',
    r'from greenlang\.benchmarks\.': 'from greenlang.extensions.benchmarks.',
    r'from greenlang\.middleware\.': 'from greenlang.extensions.middleware.',
    r'from greenlang\.satellite\.': 'from greenlang.extensions.satellite.',
    r'from greenlang\.regulations\.': 'from greenlang.extensions.regulations.',

    # Monitoring (telemetry, observability, sandbox moved into monitoring)
    r'from greenlang\.telemetry\.': 'from greenlang.monitoring.telemetry.',
    r'from greenlang\.observability\.': 'from greenlang.monitoring.observability.',
    r'from greenlang\.sandbox\.': 'from greenlang.monitoring.sandbox.',

    # Merged into existing
    r'from greenlang\.intelligence\.': 'from greenlang.agents.intelligence.',
    r'from greenlang\.calculation\.': 'from greenlang.agents.calculation.',
    r'from greenlang\.formulas\.': 'from greenlang.agents.formulas.',
    r'from greenlang\.specs\.': 'from greenlang.config.specs.',
    r'from greenlang\.registry\.': 'from greenlang.config.registry.',
    r'from greenlang\.greenlang_registry\.': 'from greenlang.config.greenlang_registry.',
    r'from greenlang\.datasets\.': 'from greenlang.data.datasets.',
    r'from greenlang\.models\.': 'from greenlang.data.models.',
    r'from greenlang\.data_engineering\.': 'from greenlang.data.data_engineering.',
    r'from greenlang\.supply_chain\.': 'from greenlang.data.supply_chain.',
    r'from greenlang\.testing\.': 'from greenlang.tests.testing.',
}

def update_file(file_path: Path, dry_run: bool = False):
    """Update imports in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  [WARN] Error reading {file_path}: {e}")
        return 0

    original_content = content
    changes = 0

    for old_pattern, new_import in IMPORT_MAPPINGS.items():
        if re.search(old_pattern, content):
            content = re.sub(old_pattern, new_import, content)
            if content != original_content:
                changes += 1
                original_content = content

    if changes > 0 and not dry_run:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return changes
        except Exception as e:
            print(f"  [WARN] Error writing {file_path}: {e}")
            return 0

    return changes

def update_imports(root_dir: Path, dry_run: bool = False):
    """Update all imports in Python files."""
    print(f"{'DRY RUN: ' if dry_run else ''}Updating imports in {root_dir}")

    total_files = 0
    total_changes = 0

    for py_file in root_dir.rglob("*.py"):
        # Skip certain directories
        if any(skip in str(py_file) for skip in ['__pycache__', '.git', 'venv', 'node_modules']):
            continue

        # Skip protected directories
        if any(protected in str(py_file) for protected in ['2026_PRD_MVP', 'cbam-pack-mvp']):
            continue

        changes = update_file(py_file, dry_run)
        if changes > 0:
            total_files += 1
            total_changes += changes
            if dry_run:
                print(f"  Would update: {py_file.relative_to(root_dir)} ({changes} changes)")
            else:
                print(f"  [OK] Updated: {py_file.relative_to(root_dir)} ({changes} changes)")

    print(f"\n{'Would update' if dry_run else 'Updated'} {total_files} files with {total_changes} import changes")
    return total_files, total_changes

if __name__ == "__main__":
    import sys

    root = Path(".")
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("=" * 60)
        print("DRY RUN MODE - No files will be modified")
        print("=" * 60 + "\n")

    total_files, total_changes = update_imports(root, dry_run)

    if not dry_run and total_changes > 0:
        print(f"\n[SUCCESS] Import updates complete!")
    elif dry_run:
        print(f"\n[INFO] Dry run complete. Run without --dry-run to apply changes.")
