#!/usr/bin/env python3
"""
Comprehensive fix for syntax errors in the GreenLang codebase.
Handles the broken import patterns more thoroughly.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Set

def fix_broken_intelligence_imports(file_path: Path) -> bool:
    """
    Fix broken intelligence imports that are split incorrectly.
    The pattern is:
    from greenlang.intelligence import (
    from greenlang.determinism import DeterministicClock
        ChatSession,
        ChatMessage,
        ...
    )

    Should be:
    from greenlang.determinism import DeterministicClock
    from greenlang.intelligence import (
        ChatSession,
        ChatMessage,
        ...
    )
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        fixed_lines = []
        i = 0
        modified = False

        while i < len(lines):
            line = lines[i]

            # Check for the broken pattern
            if 'from greenlang.intelligence import (' in line:
                # Check if next line is a determinism import (broken pattern)
                if i + 1 < len(lines) and 'from greenlang.determinism import' in lines[i + 1]:
                    # This is the broken pattern
                    # Find all the intelligence imports (they continue after the determinism line)
                    intelligence_imports = []
                    j = i + 2  # Start after the determinism import

                    # Collect the intelligence import items
                    while j < len(lines):
                        import_line = lines[j].strip()
                        if import_line == ')':
                            # End of import block
                            break
                        elif import_line and not import_line.startswith('#'):
                            intelligence_imports.append(lines[j])
                        j += 1

                    # Now reconstruct properly
                    # First add the determinism import
                    fixed_lines.append(lines[i + 1])  # The determinism import

                    # Then add the intelligence import properly
                    if intelligence_imports:
                        fixed_lines.append('from greenlang.intelligence import (\n')
                        for imp in intelligence_imports:
                            fixed_lines.append(imp)
                        fixed_lines.append(')\n')

                    # Skip all the lines we've processed
                    i = j + 1
                    modified = True
                    continue

            # Check for lines that look like orphaned import items
            elif line.strip() in ['ChatSession,', 'ChatMessage,', 'Role,', 'Budget,',
                                  'BudgetExceeded,', 'create_provider,', 'ChatSession',
                                  'ChatMessage', 'Role', 'Budget']:
                # Check if this is an orphaned import item (not inside a proper import block)
                # Look back to see if we're in an import block
                in_import = False
                for k in range(max(0, i - 5), i):
                    if 'import (' in lines[k]:
                        in_import = True
                        break

                if not in_import:
                    # This is likely an orphaned import item from our previous fix
                    # Skip it
                    i += 1
                    modified = True
                    continue

            # Check for "Fixed: Removed incomplete import" followed by orphaned items
            elif '# Fixed: Removed incomplete import' in line:
                fixed_lines.append(line)
                i += 1

                # Check if next lines are orphaned import items
                while i < len(lines):
                    next_line = lines[i].strip()
                    if next_line in ['ChatSession,', 'ChatMessage,', 'Role,', 'Budget,',
                                    'BudgetExceeded,', 'create_provider,', 'ToolRegistry,',
                                    'SystemPrompt,', ')'] or next_line.endswith(','):
                        # Skip orphaned import items
                        i += 1
                        modified = True
                    else:
                        break
                continue

            fixed_lines.append(line)
            i += 1

        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(fixed_lines)
            return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return False

def fix_indentation_after_comment(file_path: Path) -> bool:
    """
    Fix indentation errors that occur after '# Fixed:' comments.
    Remove orphaned import continuations.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        fixed_lines = []
        i = 0
        modified = False
        skip_orphans = False

        while i < len(lines):
            line = lines[i]

            # If we see the "Fixed" comment, check for orphaned imports after
            if '# Fixed: Removed incomplete import' in line:
                fixed_lines.append(line)
                i += 1
                skip_orphans = True
                continue

            # Skip orphaned import items
            if skip_orphans:
                stripped = line.strip()
                if (stripped.endswith(',') or stripped == ')') and not line.startswith('from ') and not line.startswith('import '):
                    # This looks like an orphaned import item
                    i += 1
                    modified = True
                    continue
                else:
                    skip_orphans = False

            fixed_lines.append(line)
            i += 1

        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(fixed_lines)
            return True

    except Exception as e:
        print(f"Error fixing indentation in {file_path}: {e}")

    return False

def add_missing_intelligence_imports(file_path: Path) -> bool:
    """
    Ensure intelligence imports are complete where needed.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if ChatSession, etc. are used but not imported
        needs_intelligence = False
        intelligence_items = []

        if 'ChatSession' in content and 'from greenlang.intelligence import' not in content:
            intelligence_items.append('ChatSession')
            needs_intelligence = True

        if 'ChatMessage' in content and 'ChatMessage' not in content.split('from greenlang.intelligence import')[0] if 'from greenlang.intelligence import' in content else True:
            if 'ChatMessage' not in intelligence_items:
                intelligence_items.append('ChatMessage')
                needs_intelligence = True

        if needs_intelligence and intelligence_items:
            # Add the import after determinism import
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'from greenlang.determinism import' in line:
                    # Add intelligence import after this
                    lines.insert(i + 1, f"from greenlang.intelligence import {', '.join(intelligence_items)}")
                    break

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            return True

    except Exception as e:
        print(f"Error adding intelligence imports to {file_path}: {e}")

    return False

def get_files_with_errors() -> List[Path]:
    """Get list of files that need fixing."""
    files = [
        "greenlang/agents/boiler_replacement_agent_ai.py",
        "greenlang/agents/boiler_replacement_agent_ai_v4.py",
        "greenlang/agents/carbon_agent_ai.py",
        "greenlang/agents/decarbonization_roadmap_agent_ai.py",
        "greenlang/agents/industrial_heat_pump_agent_ai.py",
        "greenlang/agents/industrial_heat_pump_agent_ai_v4.py",
        "greenlang/agents/industrial_process_heat_agent_ai.py",
        "greenlang/agents/recommendation_agent_ai.py",
        "greenlang/agents/report_agent_ai.py",
        "greenlang/agents/thermal_storage_agent_ai.py",
        "greenlang/agents/waste_heat_recovery_agent_ai.py",
        "greenlang/api/websocket/metrics_server.py",
        "greenlang/auth/permission_audit.py",
        "greenlang/auth/scim_provider.py",
        "greenlang/hub/client.py",
        "greenlang/intelligence/glrng.py",
        "greenlang/intelligence/runtime/dashboard.py",
        "greenlang/intelligence/runtime/router.py",
        "greenlang/security/signing.py",
        "greenlang/telemetry/metrics.py",
    ]
    return [Path(f) for f in files]

def add_missing_typing_imports(file_path: Path) -> bool:
    """Add missing typing imports."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        imports_needed = []

        if re.search(r'\bTuple\[', content) and 'from typing import' not in content:
            imports_needed.append('Tuple')

        if re.search(r'\bOptional\[', content) and 'Optional' not in content.split('from typing import')[0] if 'from typing import' in content else True:
            imports_needed.append('Optional')

        if re.search(r'\bAny\b', content) and 'Any' not in content.split('from typing import')[0] if 'from typing import' in content else True:
            imports_needed.append('Any')

        if imports_needed:
            lines = content.split('\n')

            # Find existing typing import
            typing_line = -1
            for i, line in enumerate(lines):
                if 'from typing import' in line:
                    typing_line = i
                    break

            if typing_line >= 0:
                # Extend existing import
                existing = lines[typing_line]
                for item in imports_needed:
                    if item not in existing:
                        existing = existing.rstrip()
                        if existing.endswith(')'):
                            existing = existing[:-1] + f', {item})'
                        else:
                            existing = existing + f', {item}'
                lines[typing_line] = existing
            else:
                # Add new import at the beginning
                insert_pos = 0
                for i, line in enumerate(lines):
                    if line.strip() and not line.startswith('#'):
                        insert_pos = i
                        break
                lines.insert(insert_pos, f"from typing import {', '.join(imports_needed)}")

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            return True

    except Exception as e:
        print(f"Error adding typing imports to {file_path}: {e}")

    return False

def add_missing_logger(file_path: Path) -> bool:
    """Add missing logger where needed."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if re.search(r'\blogger\.', content) and 'logger = logging.getLogger' not in content:
            lines = content.split('\n')

            # Check if logging is imported
            has_logging = any('import logging' in line for line in lines)

            if not has_logging:
                # Add both import and logger
                for i, line in enumerate(lines):
                    if line.strip() and not line.startswith('#'):
                        lines.insert(i, 'import logging')
                        lines.insert(i + 1, 'logger = logging.getLogger(__name__)')
                        break
            else:
                # Just add logger after logging import
                for i, line in enumerate(lines):
                    if 'import logging' in line:
                        lines.insert(i + 1, 'logger = logging.getLogger(__name__)')
                        break

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            return True

    except Exception as e:
        print(f"Error adding logger to {file_path}: {e}")

    return False

def add_missing_deterministic_imports(file_path: Path) -> bool:
    """Add missing deterministic imports."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        imports_added = []

        if 'DeterministicClock' in content and 'from greenlang.determinism import' not in content:
            imports_added.append('DeterministicClock')

        if 'deterministic_uuid' in content and 'deterministic_uuid' not in content:
            imports_added.append('deterministic_uuid')

        if imports_added:
            lines = content.split('\n')

            # Find a good place to add the import
            for i, line in enumerate(lines):
                if line.startswith('from ') or line.startswith('import '):
                    # Add after other imports
                    lines.insert(i + 1, f"from greenlang.determinism import {', '.join(imports_added)}")
                    break

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            return True

    except Exception as e:
        print(f"Error adding deterministic imports to {file_path}: {e}")

    return False

def main():
    """Main function."""
    print("Comprehensive syntax error fixing...")

    files = get_files_with_errors()

    total_fixes = 0

    for file_path in files:
        if file_path.exists():
            print(f"Processing {file_path}...")
            fixes_made = False

            # First pass: fix broken intelligence imports
            if fix_broken_intelligence_imports(file_path):
                print(f"  [FIXED] Broken intelligence imports")
                fixes_made = True

            # Second pass: fix indentation issues
            if fix_indentation_after_comment(file_path):
                print(f"  [FIXED] Indentation after comments")
                fixes_made = True

            # Third pass: add missing imports
            if add_missing_intelligence_imports(file_path):
                print(f"  [FIXED] Added missing intelligence imports")
                fixes_made = True

            if add_missing_typing_imports(file_path):
                print(f"  [FIXED] Added missing typing imports")
                fixes_made = True

            if add_missing_logger(file_path):
                print(f"  [FIXED] Added missing logger")
                fixes_made = True

            if add_missing_deterministic_imports(file_path):
                print(f"  [FIXED] Added missing deterministic imports")
                fixes_made = True

            if fixes_made:
                total_fixes += 1

    print(f"\n{'='*60}")
    print(f"Fixed {total_fixes} files")

if __name__ == "__main__":
    main()