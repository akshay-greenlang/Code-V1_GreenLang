#!/usr/bin/env python
"""
Migration script to update CLI to enhanced version
"""

import shutil
from pathlib import Path


def migrate_cli():
    """Migrate to enhanced CLI"""
    
    print("Migrating to Enhanced CLI...")
    
    # Backup original
    original = Path("greenlang/cli/main.py")
    backup = Path("greenlang/cli/main_original.py")
    
    if original.exists() and not backup.exists():
        shutil.copy2(original, backup)
        print(f"✓ Backed up original CLI to {backup}")
    
    # Copy enhanced version
    enhanced = Path("greenlang/cli/enhanced_main.py")
    if enhanced.exists():
        shutil.copy2(enhanced, original)
        print(f"✓ Installed enhanced CLI")
    
    # Update entry point if needed
    print("\nTo use the enhanced CLI, ensure pyproject.toml has:")
    print('[project.scripts]')
    print('gl = "greenlang.cli.main:cli"')
    print('gl = "greenlang.cli.main:cli"  # Short alias')
    
    print("\n✓ Migration complete!")
    print("\nNew features available:")
    print("  - Global flags: --verbose, --dry-run")
    print("  - Agent discovery: gl agents list|info|template")
    print("  - JSONL logging: gl run --log-dir")
    print("  - Report formats: gl report --format md|html|pdf|json")
    print("  - API key handling: gl ask (with helpful messages)")
    

if __name__ == "__main__":
    migrate_cli()