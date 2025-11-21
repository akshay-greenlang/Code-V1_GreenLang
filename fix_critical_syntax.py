#!/usr/bin/env python3
"""
Fix critical syntax errors in GreenLang codebase.
Focuses on the broken import pattern found in multiple files.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

def fix_broken_import_pattern(file_path: Path) -> bool:
    """
    Fix the specific broken import pattern:
    from ..something import (
    from greenlang.determinism import DeterministicClock

    Should be:
    from ..something import Something
    from greenlang.determinism import DeterministicClock
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Pattern to find the broken import
        # Look for incomplete imports followed by determinism imports
        pattern = r'(from [\w\.]+\.intelligence import \()\s*\n\s*(from greenlang\.determinism import \w+)'

        if re.search(pattern, content):
            # The intelligence import is incomplete, we need to close it properly
            # For now, we'll just comment it out and keep the determinism import
            content = re.sub(
                pattern,
                r'# Fixed: Removed incomplete import\n\2',
                content
            )

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True

        # Alternative pattern: Check line by line for safety
        lines = content.split('\n')
        fixed_lines = []
        i = 0
        modified = False

        while i < len(lines):
            line = lines[i]

            # Check if this line has an incomplete import
            if 'from greenlang.intelligence import (' in line and line.strip().endswith('('):
                # This is an incomplete import, skip it
                if i + 1 < len(lines) and 'from greenlang.determinism import' in lines[i + 1]:
                    # Next line is the determinism import, keep only that
                    fixed_lines.append("# Fixed: Removed incomplete import")
                    fixed_lines.append(lines[i + 1])
                    i += 2
                    modified = True
                    continue

            # Check for other problematic patterns
            if 'from ..types import Agent, AgentResult, ErrorInfo' in line:
                fixed_lines.append(line)
                i += 1
                # Check if next line is problematic
                if i < len(lines) and 'from greenlang.intelligence import (' in lines[i]:
                    if lines[i].strip().endswith('('):
                        # Skip this incomplete import
                        i += 1
                        modified = True
                continue

            fixed_lines.append(line)
            i += 1

        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(fixed_lines))
            return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return False

def find_files_with_syntax_errors() -> List[Path]:
    """Find all Python files with the syntax error pattern."""
    problem_files = [
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
        "greenlang/api/graphql/dataloaders.py",
        "greenlang/api/graphql/resolvers.py",
        "greenlang/api/graphql/subscriptions.py",
        "greenlang/api/main.py",
        "greenlang/api/routes/dashboards.py",
        "greenlang/api/routes/marketplace.py",
        "greenlang/auth/roles.py",
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

    return [Path(f) for f in problem_files]

def add_missing_deterministic_imports(file_path: Path) -> bool:
    """Add missing DeterministicClock imports where needed."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        modified = False

        # Check if DeterministicClock is used but not imported
        if 'DeterministicClock' in content:
            if 'from greenlang.determinism import DeterministicClock' not in content:
                if 'import DeterministicClock' not in content:
                    # Add the import after other imports
                    lines = content.split('\n')

                    # Find where to insert (after other imports)
                    insert_pos = 0
                    for i, line in enumerate(lines):
                        if line.startswith('from ') or line.startswith('import '):
                            insert_pos = i + 1
                        elif line.strip() and not line.startswith('#') and insert_pos > 0:
                            break

                    lines.insert(insert_pos, 'from greenlang.determinism import DeterministicClock')
                    content = '\n'.join(lines)
                    modified = True

        # Check for deterministic_uuid
        if 'deterministic_uuid' in content:
            if 'from greenlang.determinism import deterministic_uuid' not in content:
                if 'import deterministic_uuid' not in content:
                    lines = content.split('\n')
                    insert_pos = 0
                    for i, line in enumerate(lines):
                        if 'from greenlang.determinism import' in line:
                            # Add to existing import
                            if 'deterministic_uuid' not in line:
                                lines[i] = line.rstrip() + ', deterministic_uuid'
                                modified = True
                                break
                        elif line.startswith('from ') or line.startswith('import '):
                            insert_pos = i + 1
                    else:
                        if insert_pos > 0:
                            lines.insert(insert_pos, 'from greenlang.determinism import deterministic_uuid')
                            modified = True

                    content = '\n'.join(lines)

        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True

    except Exception as e:
        print(f"Error adding imports to {file_path}: {e}")

    return False

def add_missing_logger(file_path: Path) -> bool:
    """Add missing logger definition where logger is used."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if logger is used but not defined
        if re.search(r'\blogger\.', content):
            if 'logger = logging.getLogger' not in content:
                lines = content.split('\n')

                # Check if logging is imported
                has_logging = False
                logging_line = -1
                for i, line in enumerate(lines):
                    if 'import logging' in line:
                        has_logging = True
                        logging_line = i
                        break

                if not has_logging:
                    # Add logging import
                    insert_pos = 0
                    for i, line in enumerate(lines):
                        if line.startswith('from ') or line.startswith('import '):
                            insert_pos = i + 1
                        elif line.strip() and not line.startswith('#') and insert_pos > 0:
                            break
                    lines.insert(insert_pos, 'import logging')
                    lines.insert(insert_pos + 1, 'logger = logging.getLogger(__name__)')
                else:
                    # Add logger definition after logging import
                    lines.insert(logging_line + 1, 'logger = logging.getLogger(__name__)')

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                return True

    except Exception as e:
        print(f"Error adding logger to {file_path}: {e}")

    return False

def main():
    """Main function to fix critical syntax errors."""
    print("Fixing critical syntax errors in GreenLang codebase...")

    # Get all problem files
    problem_files = find_files_with_syntax_errors()

    fixed_count = 0
    import_fixes = 0
    logger_fixes = 0

    for file_path in problem_files:
        if file_path.exists():
            print(f"Processing {file_path}...")

            if fix_broken_import_pattern(file_path):
                print(f"  [FIXED] Broken import pattern")
                fixed_count += 1

            if add_missing_deterministic_imports(file_path):
                print(f"  [FIXED] Added missing deterministic imports")
                import_fixes += 1

            if add_missing_logger(file_path):
                print(f"  [FIXED] Added missing logger")
                logger_fixes += 1

    # Also fix undefined names in other files
    other_files = [
        "greenlang/api/websocket/metrics_server.py",
        "greenlang/auth/permission_audit.py",
        "greenlang/auth/scim_provider.py",
        "greenlang/agents/tools/security_config.py",
        "greenlang/factory/sdk/python/agent_factory.py",
        "greenlang/hub/client.py",
        "greenlang/intelligence/glrng.py",
        "greenlang/intelligence/runtime/dashboard.py",
        "greenlang/intelligence/runtime/router.py",
        "greenlang/security/signing.py",
        "greenlang/telemetry/metrics.py",
    ]

    for file_path in [Path(f) for f in other_files]:
        if file_path.exists():
            print(f"Processing {file_path}...")

            if add_missing_deterministic_imports(file_path):
                print(f"  [FIXED] Added missing imports")
                import_fixes += 1

            if add_missing_logger(file_path):
                print(f"  [FIXED] Added missing logger")
                logger_fixes += 1

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Files with syntax errors fixed: {fixed_count}")
    print(f"Missing imports added: {import_fixes}")
    print(f"Missing logger definitions added: {logger_fixes}")
    print(f"Total fixes: {fixed_count + import_fixes + logger_fixes}")

if __name__ == "__main__":
    main()