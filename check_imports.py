#!/usr/bin/env python3
"""Check for circular imports and import issues in the codebase."""

import ast
import os
from pathlib import Path
from typing import Dict, Set, List, Tuple
import json

def get_imports(file_path: Path) -> Set[str]:
    """Extract all imports from a Python file."""
    imports = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
    except (SyntaxError, UnicodeDecodeError):
        pass

    return imports

def find_cycles(graph: Dict[str, Set[str]]) -> List[List[str]]:
    """Find cycles in the import graph using DFS."""
    cycles = []
    visited = set()
    rec_stack = []

    def dfs(node: str, path: List[str]) -> None:
        if node in rec_stack:
            # Found a cycle
            cycle_start = rec_stack.index(node)
            cycle = rec_stack[cycle_start:] + [node]
            cycles.append(cycle)
            return

        if node in visited:
            return

        visited.add(node)
        rec_stack.append(node)

        for neighbor in graph.get(node, []):
            dfs(neighbor, path + [neighbor])

        rec_stack.pop()

    for node in graph:
        if node not in visited:
            dfs(node, [node])

    return cycles

def analyze_imports(root_dir: Path) -> Dict:
    """Analyze imports in the project."""
    import_graph = {}
    all_modules = set()
    issues = []

    # Scan Python files
    for py_file in root_dir.rglob("*.py"):
        if any(part.startswith('.') for part in py_file.parts):
            continue
        if 'test' in str(py_file).lower():
            continue

        module_name = str(py_file.relative_to(root_dir).with_suffix('')).replace(os.sep, '.')
        all_modules.add(module_name)

        imports = get_imports(py_file)
        filtered_imports = set()

        for imp in imports:
            # Only track internal imports
            if imp.startswith('greenlang') or imp.startswith('core.greenlang'):
                filtered_imports.add(imp)

        if filtered_imports:
            import_graph[module_name] = filtered_imports

    # Find cycles
    cycles = find_cycles(import_graph)

    # Check for problematic patterns
    for module, imports in import_graph.items():
        # Check for suspicious cross-module imports
        if 'cli' in module and any('runtime' in imp for imp in imports):
            issues.append({
                'type': 'cross_layer',
                'module': module,
                'issue': 'CLI importing from runtime layer'
            })

        if 'utils' in module and any('cli' in imp or 'hub' in imp for imp in imports):
            issues.append({
                'type': 'inverted_dependency',
                'module': module,
                'issue': 'Utils importing from higher layers'
            })

    return {
        'total_modules': len(all_modules),
        'modules_with_imports': len(import_graph),
        'cycles': cycles,
        'issues': issues,
        'import_graph_sample': dict(list(import_graph.items())[:5])
    }

if __name__ == '__main__':
    root = Path('.')
    results = analyze_imports(root)

    print("=== Import Analysis Results ===")
    print(f"Total Python modules: {results['total_modules']}")
    print(f"Modules with internal imports: {results['modules_with_imports']}")

    if results['cycles']:
        print(f"\n[WARNING] Found {len(results['cycles'])} circular dependencies:")
        for cycle in results['cycles']:
            print(f"  - {' -> '.join(cycle)}")
    else:
        print("\n[OK] No circular dependencies detected")

    if results['issues']:
        print(f"\n[WARNING] Found {len(results['issues'])} import issues:")
        for issue in results['issues']:
            print(f"  - {issue['module']}: {issue['issue']}")
    else:
        print("\n[OK] No problematic import patterns detected")

    # Save detailed results
    with open('import_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\nDetailed results saved to import_analysis.json")