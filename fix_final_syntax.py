#!/usr/bin/env python3
"""
Final pass to fix remaining syntax errors.
Removes unmatched parentheses and fixes remaining import issues.
"""

import re
from pathlib import Path
from typing import List

def remove_unmatched_parentheses(file_path: Path) -> bool:
    """Remove standalone closing parentheses that are causing syntax errors."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        fixed_lines = []
        modified = False

        for i, line in enumerate(lines):
            # Skip standalone closing parenthesis
            if line.strip() == ')':
                # Check if this is part of a valid import block
                in_import = False
                if i > 0:
                    for j in range(max(0, i - 10), i):
                        if 'import (' in lines[j] and ')' not in ''.join(lines[j:i]):
                            in_import = True
                            break

                if not in_import:
                    # This is an unmatched parenthesis, skip it
                    modified = True
                    continue

            fixed_lines.append(line)

        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(fixed_lines)
            return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return False

def fix_remaining_imports(file_path: Path) -> bool:
    """Fix any remaining broken imports."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Add missing DeterministicClock imports
        if 'DeterministicClock' in content and 'from greenlang.determinism import DeterministicClock' not in content:
            lines = content.split('\n')
            # Find a good place to insert
            for i, line in enumerate(lines):
                if line.startswith('from greenlang'):
                    lines.insert(i, 'from greenlang.determinism import DeterministicClock')
                    content = '\n'.join(lines)
                    break

        # Add missing deterministic_uuid imports
        if 'deterministic_uuid' in content and 'from greenlang.determinism import' in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'from greenlang.determinism import' in line and 'deterministic_uuid' not in line:
                    if 'DeterministicClock' in line:
                        lines[i] = line.replace('DeterministicClock', 'DeterministicClock, deterministic_uuid')
                    else:
                        lines[i] = line.rstrip() + ', deterministic_uuid'
                    content = '\n'.join(lines)
                    break

        # Write back if modified
        if content != open(file_path, 'r', encoding='utf-8').read():
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True

    except Exception as e:
        print(f"Error fixing imports in {file_path}: {e}")
    return False

def fix_specific_files() -> dict:
    """Fix specific files with known issues."""
    fixes = {
        'syntax_fixes': 0,
        'import_fixes': 0,
        'parenthesis_fixes': 0
    }

    # Files with unmatched parentheses
    paren_files = [
        "greenlang/agents/boiler_replacement_agent_ai.py",
        "greenlang/agents/carbon_agent_ai.py",
        "greenlang/agents/decarbonization_roadmap_agent_ai.py",
        "greenlang/agents/industrial_heat_pump_agent_ai.py",
        "greenlang/agents/industrial_process_heat_agent_ai.py",
        "greenlang/agents/recommendation_agent_ai.py",
        "greenlang/agents/report_agent_ai.py",
        "greenlang/agents/thermal_storage_agent_ai.py",
        "greenlang/agents/waste_heat_recovery_agent_ai.py",
    ]

    for file_str in paren_files:
        file_path = Path(file_str)
        if file_path.exists():
            if remove_unmatched_parentheses(file_path):
                fixes['parenthesis_fixes'] += 1
                print(f"Fixed unmatched parentheses in {file_path}")

    # Files needing import fixes
    import_files = [
        "greenlang/api/websocket/metrics_server.py",
        "greenlang/auth/scim_provider.py",
        "greenlang/auth/permission_audit.py",
        "greenlang/hub/client.py",
        "greenlang/security/signing.py",
        "greenlang/telemetry/metrics.py",
    ]

    for file_str in import_files:
        file_path = Path(file_str)
        if file_path.exists():
            if fix_remaining_imports(file_path):
                fixes['import_fixes'] += 1
                print(f"Fixed imports in {file_path}")

    # Fix files with syntax errors in other directories
    other_files_with_syntax = [
        "greenlang/api/graphql/dataloaders.py",
        "greenlang/api/graphql/resolvers.py",
        "greenlang/api/graphql/subscriptions.py",
        "greenlang/api/main.py",
        "greenlang/api/routes/dashboards.py",
        "greenlang/api/routes/marketplace.py",
        "greenlang/auth/roles.py",
        "greenlang/calculation/batch_calculator.py",
        "greenlang/calculation/scope1_calculator.py",
        "greenlang/calculation/scope2_calculator.py",
        "greenlang/intelligence/rag/ingest.py",
        "greenlang/intelligence/runtime/tools.py",
        "greenlang/marketplace/models.py",
        "greenlang/marketplace/monetization.py",
        "greenlang/marketplace/publisher.py",
        "greenlang/marketplace/rating_system.py",
        "greenlang/marketplace/recommendation.py",
        "greenlang/middleware/error_handler.py",
        "greenlang/runtime/backends/docker.py",
        "greenlang/runtime/backends/executor.py",
        "greenlang/runtime/backends/k8s.py",
        "greenlang/runtime/backends/local.py",
        "greenlang/services/entity_mdm/ml/evaluation.py",
        "greenlang/services/entity_mdm/ml/resolver.py",
        "greenlang/services/factor_broker/broker.py",
        "greenlang/services/factor_broker/sources/base.py",
        "greenlang/services/factor_broker/sources/desnz.py",
        "greenlang/services/factor_broker/sources/ecoinvent.py",
        "greenlang/services/factor_broker/sources/epa.py",
        "greenlang/services/factor_broker/sources/proxy.py",
        "greenlang/services/methodologies/dqi_calculator.py",
        "greenlang/services/methodologies/pedigree_matrix.py",
        "greenlang/services/pcf_exchange/catenax_client.py",
        "greenlang/services/pcf_exchange/pact_client.py",
        "greenlang/services/pcf_exchange/service.py",
    ]

    for file_str in other_files_with_syntax:
        file_path = Path(file_str)
        if file_path.exists():
            # These files likely have the same import issue pattern
            # Try to fix them by checking for misplaced imports
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                fixed_lines = []
                modified = False

                for i, line in enumerate(lines):
                    # Look for the problematic pattern
                    if i > 0 and 'from greenlang.determinism import' in line:
                        prev_line = lines[i-1]
                        # If previous line doesn't end properly, it's probably broken
                        if not (prev_line.rstrip().endswith(')') or
                               prev_line.rstrip().endswith(';') or
                               prev_line.strip().startswith('#') or
                               prev_line.strip() == ''):
                            # Skip adding this line if it seems misplaced
                            if not any(keyword in prev_line for keyword in ['from', 'import', 'class', 'def', 'if', 'for', 'while', 'try', 'except']):
                                continue

                    fixed_lines.append(line)

                if len(fixed_lines) != len(lines):
                    modified = True

                if modified:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.writelines(fixed_lines)
                    fixes['syntax_fixes'] += 1
                    print(f"Fixed syntax in {file_path}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return fixes

def add_missing_logger_imports() -> int:
    """Add missing logger imports where needed."""
    count = 0
    files_needing_logger = [
        "greenlang/hub/client.py",
        "greenlang/security/signing.py",
    ]

    for file_str in files_needing_logger:
        file_path = Path(file_str)
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                if 'logger.' in content and 'logger = logging.getLogger' not in content:
                    lines = content.split('\n')

                    # Find where to add imports
                    import_added = False
                    logger_added = False

                    for i, line in enumerate(lines):
                        if 'import ' in line and not import_added:
                            # Add logging import if not present
                            if 'import logging' not in content:
                                lines.insert(i, 'import logging')
                                import_added = True
                                i += 1

                            # Add logger definition
                            if not logger_added:
                                lines.insert(i + 1, 'logger = logging.getLogger(__name__)')
                                logger_added = True
                                break

                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(lines))
                    count += 1
                    print(f"Added logger to {file_path}")

            except Exception as e:
                print(f"Error adding logger to {file_path}: {e}")

    return count

def main():
    """Main function."""
    print("Final syntax error fixes...")
    print("="*60)

    # Fix specific issues
    fixes = fix_specific_files()

    # Add missing loggers
    logger_fixes = add_missing_logger_imports()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Parenthesis fixes: {fixes['parenthesis_fixes']}")
    print(f"Import fixes: {fixes['import_fixes']}")
    print(f"Syntax fixes: {fixes['syntax_fixes']}")
    print(f"Logger additions: {logger_fixes}")
    print(f"Total fixes: {sum(fixes.values()) + logger_fixes}")

if __name__ == "__main__":
    main()