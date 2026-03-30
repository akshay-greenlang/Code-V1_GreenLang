#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared Schema Migration Script
===============================
Migrates agent model files from raw `pydantic.BaseModel` to GreenLang shared
base classes (`greenlang.schemas`).

Migration rules:
  1. Replace `from pydantic import BaseModel, ...` -> keep Field/validators,
     add `from greenlang.schemas import GreenLangBase, utcnow, new_uuid`
  2. Replace `class Foo(BaseModel):` -> `class Foo(GreenLangBase):`
  3. Remove redundant `model_config = ConfigDict(extra="forbid")` (GreenLangBase provides it)
  4. If file already partially imports from greenlang.schemas, merge imports

Author: GreenLang Platform Team
Date: 2026-03-30
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Files under greenlang/agents/ that still use `from pydantic import ... BaseModel`
# and need migration to shared base classes.
MRV_MODELS = [
    "greenlang/agents/mrv/capital_goods/models.py",
    "greenlang/agents/mrv/upstream_transportation/models.py",
    "greenlang/agents/mrv/steam_heat_purchase/models.py",
    "greenlang/agents/mrv/refrigerants_fgas/models.py",
    "greenlang/agents/mrv/process_emissions/models.py",
    "greenlang/agents/mrv/fugitive_emissions/models.py",
    "greenlang/agents/mrv/cooling_purchase/models.py",
    "greenlang/agents/mrv/waste_treatment_emissions/models.py",
    "greenlang/agents/mrv/scope2_market/models.py",
    "greenlang/agents/mrv/scope2_location/models.py",
    "greenlang/agents/mrv/purchased_goods_services/models.py",
    "greenlang/agents/mrv/mobile_combustion/models.py",
    "greenlang/agents/mrv/land_use_emissions/models.py",
    "greenlang/agents/mrv/fuel_energy_activities/models.py",
    "greenlang/agents/mrv/flaring/models.py",
    "greenlang/agents/mrv/dual_reporting_reconciliation/models.py",
    "greenlang/agents/mrv/scope3_category_mapper/models.py",
    "greenlang/agents/mrv/investments/models.py",
    "greenlang/agents/mrv/franchises/models.py",
    "greenlang/agents/mrv/downstream_leased_assets/models.py",
    "greenlang/agents/mrv/end_of_life_treatment/models.py",
    "greenlang/agents/mrv/use_of_sold_products/models.py",
    "greenlang/agents/mrv/processing_sold_products/models.py",
    "greenlang/agents/mrv/downstream_transportation/models.py",
    "greenlang/agents/mrv/upstream_leased_assets/models.py",
    "greenlang/agents/mrv/employee_commuting/models.py",
    "greenlang/agents/mrv/waste_generated/models.py",
    # Already partially migrated (have both imports) - still need BaseModel replaced
    "greenlang/agents/mrv/stationary_combustion/models.py",
    "greenlang/agents/data/pdf_extractor/models.py",
]

# Additional non-MRV agent models
OTHER_MODELS = [
    "greenlang/agents/formulas/models.py",
    "greenlang/agents/intelligence/rag/models.py",
]


def migrate_file(filepath: Path, dry_run: bool = False) -> dict:
    """Migrate a single model file to use shared schema base classes.

    Returns dict with migration stats.
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
    if "from pydantic import" not in content or "BaseModel" not in content:
        # Check if it's a pure greenlang.schemas user
        if "from greenlang.schemas import" in content and "BaseModel" not in content:
            stats["already_migrated"] = True
            return stats

    # -----------------------------------------------------------------------
    # Step 1: Update pydantic import line - remove BaseModel and ConfigDict
    # -----------------------------------------------------------------------
    # Match patterns like:
    #   from pydantic import BaseModel, ConfigDict, Field, validator
    #   from pydantic import BaseModel, Field, validator, field_validator, model_validator
    #   from pydantic import BaseModel, ConfigDict, Field, field_validator

    pydantic_import_pattern = re.compile(
        r'^(from pydantic import\s+)(.*?)$',
        re.MULTILINE
    )

    def fix_pydantic_import(match):
        prefix = match.group(1)
        imports_str = match.group(2)

        # Parse the imports
        imports = [i.strip() for i in imports_str.split(",")]
        imports = [i for i in imports if i]  # remove empty

        # Remove BaseModel and ConfigDict (provided by GreenLangBase)
        remaining = [i for i in imports if i not in ("BaseModel", "ConfigDict")]

        if remaining:
            return prefix + ", ".join(remaining)
        else:
            return ""  # All imports were BaseModel/ConfigDict

    content = pydantic_import_pattern.sub(fix_pydantic_import, content)

    # Remove standalone `from pydantic import ConfigDict` lines
    content = re.sub(
        r'^from pydantic import ConfigDict\s*$\n?',
        '',
        content,
        flags=re.MULTILINE
    )

    # Remove empty pydantic import lines that might result
    content = re.sub(
        r'^from pydantic import\s*$\n?',
        '',
        content,
        flags=re.MULTILINE
    )

    stats["import_updated"] = True

    # -----------------------------------------------------------------------
    # Step 2: Add/update greenlang.schemas import
    # -----------------------------------------------------------------------
    # Check if there's already a greenlang.schemas import
    has_schemas_import = "from greenlang.schemas import" in content or \
                         "from greenlang.schemas" in content

    # Determine what to import from greenlang.schemas
    needed_imports = ["GreenLangBase"]

    # Check if utcnow is used but not imported from schemas
    if "utcnow" in content and "from greenlang.schemas import utcnow" not in content \
            and "from greenlang.schemas import" not in content:
        needed_imports.append("utcnow")

    if has_schemas_import:
        # Already has a schemas import - add GreenLangBase if not there
        if "GreenLangBase" not in content:
            # Add GreenLangBase to existing import
            content = re.sub(
                r'(from greenlang\.schemas import\s+)(.*?)$',
                lambda m: m.group(1) + "GreenLangBase, " + m.group(2)
                    if "GreenLangBase" not in m.group(2) else m.group(0),
                content,
                count=1,
                flags=re.MULTILINE
            )
    else:
        # No schemas import - add one after the pydantic import or at the import section
        import_line = "from greenlang.schemas import GreenLangBase, utcnow, new_uuid\n"

        # Find a good place to insert - after the last pydantic import
        pydantic_pos = content.rfind("from pydantic import")
        if pydantic_pos >= 0:
            # Find end of that line
            eol = content.find("\n", pydantic_pos)
            if eol >= 0:
                content = content[:eol + 1] + import_line + content[eol + 1:]
        else:
            # No pydantic import left - insert after other imports
            # Find the last import line
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
    # since GreenLangBase already sets this.
    # -----------------------------------------------------------------------
    # Match various forms:
    #   model_config = ConfigDict(extra="forbid")
    #   model_config = ConfigDict(extra="forbid", frozen=True)
    # Only remove if it's exactly extra="forbid" with no other settings
    configdict_pattern = re.compile(
        r'^(\s*)model_config\s*=\s*ConfigDict\(\s*extra\s*=\s*["\']forbid["\']\s*\)\s*$\n?',
        re.MULTILINE
    )
    removed = len(configdict_pattern.findall(content))
    content = configdict_pattern.sub('', content)
    stats["configdict_removed"] = removed

    # -----------------------------------------------------------------------
    # Step 5: Clean up - remove duplicate blank lines
    # -----------------------------------------------------------------------
    content = re.sub(r'\n{4,}', '\n\n\n', content)

    # -----------------------------------------------------------------------
    # Write result
    # -----------------------------------------------------------------------
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

    all_files = MRV_MODELS + OTHER_MODELS

    total_replaced = 0
    total_configdict = 0
    migrated = 0
    skipped = 0
    errors = 0

    print(f"{'DRY RUN - ' if dry_run else ''}Shared Schema Migration")
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
