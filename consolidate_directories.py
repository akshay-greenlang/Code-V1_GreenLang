#!/usr/bin/env python3
"""
Directory consolidation script for GreenLang.
Reduces greenlang/ subdirectories from 77 to ≤15.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List

# Base directory
GREENLANG_DIR = Path("greenlang")

# Consolidation mapping: source_dir -> target_dir
CONSOLIDATION_MAP = {
    # Remove empty/near-empty directories (will be deleted)
    "calculators": None,  # DELETE - only __init__.py
    "docs": None,  # DELETE - empty
    "emission_factors": None,  # DELETE - empty
    "frontend": None,  # DELETE - empty
    "schemas": None,  # DELETE - empty
    "src": None,  # DELETE - empty
    "observability": "monitoring",  # MERGE - only 1 file, merge into monitoring

    # Database consolidation (database is already deprecated, pointing to db)
    "database": None,  # DELETE - already deprecated wrapper for db
    "datasets": "data",  # MERGE - dataset loaders go into data
    "models": "data",  # MERGE - emission factor models go into data

    # Registry consolidation
    "greenlang_registry": "registry",  # MERGE - consolidate registries

    # API/Integration consolidation
    "connectors": "integrations",  # MERGE - connectors are part of integrations
    "services": "integrations",  # MERGE - services layer into integrations
    "adapters": "integrations",  # MERGE - adapters into integrations
    "sdk": "integration",  # MERGE - SDK into new integration module
    "api": "integration",  # MERGE - API into new integration module

    # Execution consolidation
    "pipeline": "execution",  # MERGE - pipelines into new execution module
    "runtime": "execution",  # MERGE - runtime into execution
    "infrastructure": "execution",  # MERGE - infra into execution
    "core": "execution",  # MERGE - core execution logic
    "resilience": "execution",  # MERGE - resilience patterns into execution

    # Utilities consolidation
    "io": "utilities",  # MERGE - I/O utilities
    "serialization": "utilities",  # MERGE - serialization utilities
    "determinism": "utilities",  # MERGE - determinism utilities
    "lineage": "utilities",  # MERGE - lineage tracking
    "tools": "utilities",  # MERGE - misc tools
    "visualization": "utilities",  # MERGE - visualization tools
    "cards": "utilities",  # MERGE - card generation
    "factory": "utilities",  # MERGE - factory patterns
    "generator": "utilities",  # MERGE - code generation
    "compat": "utilities",  # MERGE - compatibility utilities
    "i18n": "utilities",  # MERGE - internationalization
    "exceptions": "utilities",  # MERGE - exception classes

    # Monitoring/Telemetry consolidation
    "telemetry": "monitoring",  # MERGE - telemetry into monitoring
    "sandbox": "monitoring",  # MERGE - sandbox monitoring

    # Security/Governance consolidation
    "policy": "governance",  # MERGE - policies into new governance module
    "safety": "governance",  # MERGE - safety into governance
    "compliance": "governance",  # MERGE - compliance into governance
    "security": "governance",  # MERGE - security into governance

    # Ecosystem consolidation
    "marketplace": "ecosystem",  # MERGE - marketplace into new ecosystem module
    "packs": "ecosystem",  # MERGE - packs into ecosystem
    "hub": "ecosystem",  # MERGE - hub into ecosystem
    "partners": "ecosystem",  # MERGE - partners into ecosystem
    "whitelabel": "ecosystem",  # MERGE - whitelabel into ecosystem

    # Extensions consolidation
    "ml": "extensions",  # MERGE - ML into new extensions module
    "ml_platform": "extensions",  # MERGE - ML platform into extensions
    "llm": "extensions",  # MERGE - LLM into extensions
    "simulation": "extensions",  # MERGE - simulation into extensions
    "business": "extensions",  # MERGE - business logic into extensions
    "benchmarks": "extensions",  # MERGE - benchmarks into extensions
    "middleware": "extensions",  # MERGE - middleware into extensions
    "satellite": "extensions",  # MERGE - satellite into extensions
    "regulations": "extensions",  # MERGE - regulations into extensions

    # Testing consolidation
    "testing": "tests",  # MERGE - testing utilities into tests
    "examples": "tests",  # MERGE - examples can be test examples
    "templates": "tests",  # MERGE - test templates

    # Further consolidation to get to ≤15
    "utils": "utilities",  # MERGE - utils into utilities
    "provenance": "utilities",  # MERGE - provenance is a utility
    "intelligence": "agents",  # MERGE - intelligence is part of agents
    "specs": "config",  # MERGE - specs are configuration
    "cache": "utilities",  # MERGE - cache is a utility
    "validation": "governance",  # MERGE - validation is governance
    "registry": "config",  # MERGE - registry is configuration

    # Keep these as-is (no mapping needed)
    # - agents (includes intelligence now)
    # - auth
    # - cli
    # - config (includes specs, registry)
    # - data
    # - db (consolidated database layer)
    # - tests
    # New consolidated directories:
    # - integration (api, sdk, connectors, services, adapters)
    # - execution (core, pipeline, runtime, infrastructure, resilience)
    # - governance (policy, safety, compliance, security, validation)
    # - utilities (utils, io, serialization, determinism, lineage, provenance, cache, etc.)
    # - ecosystem (marketplace, packs, hub, partners, whitelabel)
    # - extensions (ml, llm, simulation, business, benchmarks, middleware, satellite, regulations)
    # - monitoring (telemetry, observability, sandbox)
}

def count_python_files(directory: Path) -> int:
    """Count Python files in a directory."""
    if not directory.exists():
        return 0
    return len(list(directory.rglob("*.py")))

def create_backward_compat_stub(old_path: Path, new_location: str):
    """Create a backward compatibility stub with deprecation warning."""
    module_name = old_path.name

    stub_content = f'''"""
DEPRECATED: {module_name} module

This module has been consolidated into greenlang.{new_location}.
This file provides backward-compatible re-exports with deprecation warnings.

Please update your imports:
    OLD: from greenlang.{module_name} import ...
    NEW: from greenlang.{new_location} import ...

This compatibility layer will be removed in version 2.0.0.
"""

import warnings

warnings.warn(
    f"greenlang.{module_name} is deprecated. "
    f"Import from greenlang.{new_location} instead. "
    "This compatibility layer will be removed in version 2.0.0.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the new location
try:
    from greenlang.{new_location}.{module_name} import *
except ImportError:
    # If the submodule doesn't exist, try importing from the parent
    from greenlang.{new_location} import *

__all__ = []
__version__ = '1.0.0-deprecated'
'''

    init_file = old_path / "__init__.py"
    with open(init_file, 'w') as f:
        f.write(stub_content)

def merge_directory(source: Path, target: Path, create_stub: bool = True):
    """Merge source directory into target directory."""
    if not source.exists():
        print(f"  ⚠ Source {source.name} doesn't exist, skipping")
        return

    if not target.exists():
        print(f"  📁 Creating new target directory: {target.name}")
        target.mkdir(parents=True, exist_ok=True)

    # Create a subdirectory in target with the source name
    target_subdir = target / source.name

    print(f"  📦 Moving {source.name} -> {target.name}/{source.name}")

    # Move the entire directory as a subdirectory
    if target_subdir.exists():
        print(f"    Merging into existing {target_subdir.name}")
        # Merge contents
        for item in source.iterdir():
            if item.name == "__pycache__":
                continue
            dest = target_subdir / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.move(str(item), str(dest))
            else:
                shutil.copy2(item, dest)
    else:
        shutil.move(str(source), str(target_subdir))

    # Create backward compatibility stub
    if create_stub and not source.exists():
        source.mkdir(parents=True, exist_ok=True)
        create_backward_compat_stub(source, target.name)
        print(f"  ✓ Created compatibility stub at {source.name}/")

def delete_directory(directory: Path):
    """Delete an empty or deprecated directory."""
    if not directory.exists():
        print(f"  ⚠ {directory.name} doesn't exist, skipping")
        return

    file_count = count_python_files(directory)
    print(f"  🗑 Deleting {directory.name} ({file_count} Python files)")

    # Keep a stub for backward compatibility
    if file_count > 1:
        print(f"    ⚠ WARNING: Deleting directory with {file_count} files!")

    shutil.rmtree(directory)

    # Create minimal stub
    directory.mkdir(parents=True, exist_ok=True)
    stub_content = '''"""
DEPRECATED: This module has been removed or consolidated.

This module is deprecated and its functionality has been moved.
Please consult CONSOLIDATION_PLAN.md for migration guidance.

This compatibility stub will be removed in version 2.0.0.
"""

import warnings

warnings.warn(
    "This module is deprecated and has been removed.",
    DeprecationWarning,
    stacklevel=2
)

__version__ = '1.0.0-deprecated'
'''
    with open(directory / "__init__.py", 'w') as f:
        f.write(stub_content)

def consolidate():
    """Execute the consolidation plan."""
    print("🚀 Starting GreenLang directory consolidation")
    print(f"📊 Target: Reduce from 77 to ≤15 subdirectories\n")

    # Group by target directory
    consolidations: Dict[str, List[str]] = {}
    deletions: List[str] = []

    for source, target in CONSOLIDATION_MAP.items():
        if target is None:
            deletions.append(source)
        else:
            if target not in consolidations:
                consolidations[target] = []
            consolidations[target].append(source)

    # Phase 1: Deletions
    print("=" * 60)
    print("PHASE 1: Removing empty/deprecated directories")
    print("=" * 60)
    for dir_name in deletions:
        dir_path = GREENLANG_DIR / dir_name
        delete_directory(dir_path)

    # Phase 2: Consolidations
    print("\n" + "=" * 60)
    print("PHASE 2: Consolidating related directories")
    print("=" * 60)
    for target, sources in sorted(consolidations.items()):
        print(f"\n📂 Consolidating into: {target}/")
        target_path = GREENLANG_DIR / target
        for source in sources:
            source_path = GREENLANG_DIR / source
            merge_directory(source_path, target_path)

    # Phase 3: Summary
    print("\n" + "=" * 60)
    print("PHASE 3: Summary")
    print("=" * 60)

    remaining_dirs = [d for d in GREENLANG_DIR.iterdir()
                     if d.is_dir() and not d.name.startswith('__')]

    print(f"\n✅ Remaining directories: {len(remaining_dirs)}")
    print("Directories kept/created:")
    for d in sorted(remaining_dirs, key=lambda x: x.name):
        file_count = count_python_files(d)
        print(f"  - {d.name:30s} ({file_count:3d} Python files)")

    if len(remaining_dirs) <= 15:
        print(f"\n🎉 SUCCESS! Reduced to {len(remaining_dirs)} directories (target: ≤15)")
    else:
        print(f"\n⚠ Still have {len(remaining_dirs)} directories (target: ≤15)")
        print("  Further consolidation needed.")

if __name__ == "__main__":
    import sys

    if "--dry-run" in sys.argv:
        print("DRY RUN MODE - No changes will be made")
        print("\nConsolidation plan:")
        for source, target in sorted(CONSOLIDATION_MAP.items()):
            if target is None:
                print(f"  DELETE: {source}")
            else:
                print(f"  MERGE:  {source} -> {target}")
    else:
        confirm = input("This will reorganize the greenlang/ directory. Continue? [y/N]: ")
        if confirm.lower() == 'y':
            consolidate()
        else:
            print("Cancelled.")
