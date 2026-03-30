#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared Schema Migration - Application Models
=============================================
Migrates application-level model files from raw pydantic.BaseModel
to greenlang.schemas.GreenLangBase.

Author: GreenLang Platform Team
Date: 2026-03-30
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

APP_MODELS = [
    # GL-Agent-Factory
    "applications/GL-Agent-Factory/backend/agents/gl_022_superheater_control/models.py",
    "applications/GL-Agent-Factory/backend/agents/gl_023_heat_load_balancer/models.py",
    "applications/GL-Agent-Factory/backend/agents/gl_031_furnace_guardian/models.py",
    "applications/GL-Agent-Factory/backend/agents/gl_032_refractory_monitor/models.py",
    "applications/GL-Agent-Factory/backend/agents/gl_033_burner_balancer/models.py",
    # GL-VCCI-Carbon-APP
    "applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/services/agents/calculator/models.py",
    "applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/services/agents/engagement/models.py",
    "applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/services/agents/hotspot/models.py",
    "applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/services/agents/intake/models.py",
    "applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/services/agents/reporting/models.py",
    # GreenLang Development copies
    "GreenLang Development/02-Applications/GL-Agent-Factory/backend/agents/gl_022_superheater_control/models.py",
    "GreenLang Development/02-Applications/GL-Agent-Factory/backend/agents/gl_023_heat_load_balancer/models.py",
    "GreenLang Development/02-Applications/GL-Agent-Factory/backend/agents/gl_031_furnace_guardian/models.py",
    "GreenLang Development/02-Applications/GL-Agent-Factory/backend/agents/gl_032_refractory_monitor/models.py",
    "GreenLang Development/02-Applications/GL-Agent-Factory/backend/agents/gl_033_burner_balancer/models.py",
    "GreenLang Development/02-Applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/services/agents/calculator/models.py",
    "GreenLang Development/02-Applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/services/agents/engagement/models.py",
    "GreenLang Development/02-Applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/services/agents/hotspot/models.py",
    "GreenLang Development/02-Applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/services/agents/intake/models.py",
    "GreenLang Development/02-Applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/services/agents/reporting/models.py",
    "GreenLang Development/01-Core-Platform/agents/formulas/models.py",
    "GreenLang Development/01-Core-Platform/agents/intelligence/rag/models.py",
]


def migrate_file(filepath: Path, dry_run: bool = False) -> dict:
    """Migrate a single model file to use shared schema base classes."""
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

    if "BaseModel" not in content:
        stats["already_migrated"] = True
        return stats

    # Handle multi-line pydantic imports
    pydantic_block_pattern = re.compile(
        r'from pydantic import \((.*?)\)',
        re.DOTALL,
    )

    def fix_pydantic_block(match):
        block = match.group(1)
        items = [item.strip().rstrip(",") for item in block.split("\n")]
        items = [item for item in items if item and not item.startswith("#")]
        remaining = [item for item in items if item not in ("BaseModel", "ConfigDict")]
        if not remaining:
            return ""
        elif len(remaining) == 1:
            return f"from pydantic import {remaining[0]}"
        else:
            inner = ",\n    ".join(remaining)
            return f"from pydantic import (\n    {inner},\n)"

    content = pydantic_block_pattern.sub(fix_pydantic_block, content)

    # Handle single-line imports
    single_line_pattern = re.compile(
        r'^from pydantic import\s+(.+)$',
        re.MULTILINE,
    )

    def fix_single_line(match):
        imports_str = match.group(1)
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
    content = re.sub(r'^from pydantic import\s*$\n?', '', content, flags=re.MULTILINE)

    stats["import_updated"] = True

    # Add GreenLangBase import
    has_schemas_import = "from greenlang.schemas import" in content
    if has_schemas_import:
        if "GreenLangBase" not in content:
            content = re.sub(
                r'(from greenlang\.schemas import\s+)(.*?)$',
                lambda m: m.group(1) + "GreenLangBase, " + m.group(2)
                if "GreenLangBase" not in m.group(2) else m.group(0),
                content,
                count=1,
                flags=re.MULTILINE,
            )
    else:
        import_line = "from greenlang.schemas import GreenLangBase, utcnow, new_uuid\n"
        pydantic_pos = content.rfind("from pydantic import")
        if pydantic_pos >= 0:
            eol = content.find("\n", pydantic_pos)
            if eol >= 0:
                content = content[:eol + 1] + import_line + content[eol + 1:]
        else:
            last_import = 0
            for m in re.finditer(r'^(?:from|import)\s+\S+', content, re.MULTILINE):
                last_import = m.end()
            eol = content.find("\n", last_import)
            if eol >= 0:
                content = content[:eol + 1] + "\n" + import_line + content[eol + 1:]

    # Replace BaseModel with GreenLangBase in class definitions
    class_pattern = re.compile(r'class\s+(\w+)\(BaseModel\):')
    count = len(class_pattern.findall(content))
    content = class_pattern.sub(r'class \1(GreenLangBase):', content)
    stats["basemodel_replaced"] = count

    # Remove redundant ConfigDict
    configdict_pattern = re.compile(
        r'^(\s*)model_config\s*=\s*ConfigDict\(\s*extra\s*=\s*["\']forbid["\']\s*\)\s*$\n?',
        re.MULTILINE,
    )
    removed = len(configdict_pattern.findall(content))
    content = configdict_pattern.sub('', content)
    stats["configdict_removed"] = removed

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

    total_replaced = 0
    migrated = 0
    skipped = 0
    errors = 0

    print(f"{'DRY RUN - ' if dry_run else ''}Application Models Schema Migration")
    print(f"{'=' * 60}")
    print(f"Files to process: {len(APP_MODELS)}")
    print()

    for rel_path in APP_MODELS:
        filepath = ROOT / rel_path
        stats = migrate_file(filepath, dry_run=dry_run)

        if stats.get("error"):
            print(f"  ERROR  {stats['file']}: {stats['error']}")
            errors += 1
        elif stats.get("already_migrated"):
            print(f"  SKIP   {stats['file']}")
            skipped += 1
        else:
            print(f"  DONE   {stats['file']}: {stats['basemodel_replaced']} classes")
            total_replaced += stats["basemodel_replaced"]
            migrated += 1

    print()
    print(f"{'=' * 60}")
    print(f"  Migrated: {migrated} | Skipped: {skipped} | Errors: {errors}")
    print(f"  Classes:  {total_replaced} BaseModel -> GreenLangBase")
    if dry_run:
        print(f"\n  DRY RUN - Remove --dry-run to apply.")


if __name__ == "__main__":
    main()
