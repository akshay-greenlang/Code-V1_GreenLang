#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared Schema Migration Script v2
===================================
Handles multi-line pydantic imports (parenthesized) and EUDR agent models.

Migrates agent model files from raw `pydantic.BaseModel` to GreenLang shared
base classes (`greenlang.schemas.GreenLangBase`).

Author: GreenLang Platform Team
Date: 2026-03-30
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# EUDR agent models with multi-line pydantic imports containing BaseModel
EUDR_MODELS = [
    "greenlang/agents/eudr/country_risk_evaluator/models.py",
    "greenlang/agents/eudr/chain_of_custody/models.py",
    "greenlang/agents/eudr/third_party_audit_manager/models.py",
    "greenlang/agents/eudr/corruption_index_monitor/models.py",
    "greenlang/agents/eudr/commodity_risk_analyzer/models.py",
    "greenlang/agents/eudr/blockchain_integration/models.py",
    "greenlang/agents/eudr/supply_chain_mapper/models.py",
    "greenlang/agents/eudr/land_use_change/models.py",
    "greenlang/agents/eudr/supplier_risk_scorer/models.py",
    "greenlang/agents/eudr/due_diligence_orchestrator/models.py",
    "greenlang/agents/eudr/document_authentication/models.py",
    "greenlang/agents/eudr/geolocation_verification/models.py",
    "greenlang/agents/eudr/segregation_verifier/models.py",
    "greenlang/agents/eudr/forest_cover_analysis/models.py",
    "greenlang/agents/eudr/deforestation_alert_system/models.py",
    "greenlang/agents/eudr/satellite_monitoring/models.py",
    "greenlang/agents/eudr/gps_coordinate_validator/models.py",
    "greenlang/agents/eudr/risk_mitigation_advisor/models.py",
    "greenlang/agents/eudr/qr_code_generator/models.py",
    "greenlang/agents/eudr/multi_tier_supplier/models.py",
    "greenlang/agents/eudr/protected_area_validator/models.py",
    "greenlang/agents/eudr/mobile_data_collector/models.py",
    "greenlang/agents/eudr/plot_boundary/models.py",
    "greenlang/agents/eudr/mass_balance_calculator/models.py",
    # DATA agent that also has BaseModel classes
    "greenlang/agents/data/eudr_traceability/models.py",
]


def migrate_file(filepath: Path, dry_run: bool = False) -> dict:
    """Migrate a single model file to use shared schema base classes.

    Handles both single-line and multi-line (parenthesized) pydantic imports.
    """
    stats = {
        "file": str(filepath.relative_to(ROOT)),
        "basemodel_replaced": 0,
        "configdict_removed": 0,
        "import_updated": False,
        "already_migrated": False,
        "error": None,
    }

    if not filepath.exists():
        stats["error"] = "File not found"
        return stats

    content = filepath.read_text(encoding="utf-8")
    original = content

    # Check if file uses BaseModel at all
    if "BaseModel" not in content:
        stats["already_migrated"] = True
        return stats

    # -----------------------------------------------------------------------
    # Step 1: Handle multi-line pydantic imports
    # -----------------------------------------------------------------------
    # Match: from pydantic import (\n    BaseModel,\n    ConfigDict,\n    Field,\n)
    pydantic_block_pattern = re.compile(
        r'from pydantic import \((.*?)\)',
        re.DOTALL
    )

    def fix_pydantic_block(match):
        block = match.group(1)
        # Parse individual imports from the block
        items = [item.strip().rstrip(",") for item in block.split("\n")]
        items = [item for item in items if item and not item.startswith("#")]

        # Remove BaseModel and ConfigDict
        remaining = [item for item in items if item not in ("BaseModel", "ConfigDict")]

        if not remaining:
            return ""  # All items removed
        elif len(remaining) == 1:
            return f"from pydantic import {remaining[0]}"
        else:
            inner = ",\n    ".join(remaining)
            return f"from pydantic import (\n    {inner},\n)"

    content = pydantic_block_pattern.sub(fix_pydantic_block, content)

    # Also handle single-line imports
    single_line_pattern = re.compile(
        r'^from pydantic import\s+(.+)$',
        re.MULTILINE
    )

    def fix_single_line(match):
        imports_str = match.group(1)
        # Skip if it's a parenthesized import (already handled)
        if imports_str.startswith("("):
            return match.group(0)
        imports = [i.strip() for i in imports_str.split(",")]
        imports = [i for i in imports if i]
        remaining = [i for i in imports if i not in ("BaseModel", "ConfigDict")]
        if remaining:
            return f"from pydantic import {', '.join(remaining)}"
        else:
            return ""

    content = single_line_pattern.sub(fix_single_line, content)

    # Remove empty pydantic import lines
    content = re.sub(r'^from pydantic import\s*$\n?', '', content, flags=re.MULTILINE)

    stats["import_updated"] = True

    # -----------------------------------------------------------------------
    # Step 2: Add GreenLangBase to greenlang.schemas import if not present
    # -----------------------------------------------------------------------
    has_schemas_import = "from greenlang.schemas import" in content

    if has_schemas_import:
        if "GreenLangBase" not in content:
            # Add GreenLangBase to existing import
            content = re.sub(
                r'(from greenlang\.schemas import\s+)(.*?)$',
                lambda m: m.group(1) + "GreenLangBase, " + m.group(2)
                if "GreenLangBase" not in m.group(2) else m.group(0),
                content,
                count=1,
                flags=re.MULTILINE,
            )
    else:
        # Need to add a new greenlang.schemas import line
        import_line = "from greenlang.schemas import GreenLangBase, utcnow, new_uuid\n"

        # Find good insertion point - after last pydantic import or after other imports
        pydantic_pos = content.rfind("from pydantic import")
        if pydantic_pos >= 0:
            eol = content.find("\n", pydantic_pos)
            if eol >= 0:
                content = content[:eol + 1] + import_line + content[eol + 1:]
        else:
            # After last import statement
            last_import = 0
            for m in re.finditer(r'^(?:from|import)\s+\S+', content, re.MULTILINE):
                last_import = m.end()
            eol = content.find("\n", last_import)
            if eol >= 0:
                content = content[:eol + 1] + "\n" + import_line + content[eol + 1:]

    # -----------------------------------------------------------------------
    # Step 3: Replace class Foo(BaseModel): with class Foo(GreenLangBase):
    # -----------------------------------------------------------------------
    class_pattern = re.compile(r'class\s+(\w+)\(BaseModel\):')
    count = len(class_pattern.findall(content))
    content = class_pattern.sub(r'class \1(GreenLangBase):', content)
    stats["basemodel_replaced"] = count

    # -----------------------------------------------------------------------
    # Step 4: Remove redundant model_config = ConfigDict(extra="forbid")
    # -----------------------------------------------------------------------
    configdict_pattern = re.compile(
        r'^(\s*)model_config\s*=\s*ConfigDict\(\s*extra\s*=\s*["\']forbid["\']\s*\)\s*$\n?',
        re.MULTILINE,
    )
    removed = len(configdict_pattern.findall(content))
    content = configdict_pattern.sub('', content)
    stats["configdict_removed"] = removed

    # -----------------------------------------------------------------------
    # Step 5: Clean up multiple blank lines
    # -----------------------------------------------------------------------
    content = re.sub(r'\n{4,}', '\n\n\n', content)

    if content != original:
        if not dry_run:
            filepath.write_text(content, encoding="utf-8")
        return stats
    else:
        stats["already_migrated"] = True
        return stats


def main():
    dry_run = "--dry-run" in sys.argv
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    all_files = EUDR_MODELS

    total_replaced = 0
    total_configdict = 0
    migrated = 0
    skipped = 0
    errors = 0

    print(f"{'DRY RUN - ' if dry_run else ''}Shared Schema Migration v2 (EUDR + multi-line)")
    print(f"{'=' * 60}")
    print(f"Files to process: {len(all_files)}")
    print()

    for rel_path in all_files:
        filepath = ROOT / rel_path.replace("/", "\\")
        stats = migrate_file(filepath, dry_run=dry_run)

        if stats.get("error"):
            print(f"  ERROR  {stats['file']}: {stats['error']}")
            errors += 1
        elif stats.get("already_migrated"):
            if verbose:
                print(f"  SKIP   {stats['file']} (already migrated)")
            skipped += 1
        else:
            print(f"  DONE   {stats['file']}: "
                  f"{stats['basemodel_replaced']} classes, "
                  f"{stats['configdict_removed']} ConfigDict removed")
            total_replaced += stats["basemodel_replaced"]
            total_configdict += stats["configdict_removed"]
            migrated += 1

    print()
    print(f"{'=' * 60}")
    print(f"Summary:")
    print(f"  Migrated:    {migrated} files")
    print(f"  Skipped:     {skipped} files (already done)")
    print(f"  Errors:      {errors} files")
    print(f"  Classes:     {total_replaced} BaseModel -> GreenLangBase")
    print(f"  ConfigDict:  {total_configdict} redundant entries removed")
    if dry_run:
        print(f"\n  DRY RUN - no files were modified. Remove --dry-run to apply.")


if __name__ == "__main__":
    main()
