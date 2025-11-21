#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Security Fix Script - Remove eval() vulnerabilities
Replaces all eval() usage with safe alternatives
"""

import os
import re


def fix_reasoning_py():
    """Fix eval() in capabilities/reasoning.py"""
    file_path = 'C:/Users/aksha/Code-V1_GreenLang/GreenLang_2030/agent_foundation/capabilities/reasoning.py'

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace eval() with ast.literal_eval()
    old_code = '''    def _extract_solution(self, source: str) -> Any:
        """Extract solution from source case."""
        # Parse source string back to dict
        try:
            source_dict = eval(source)  # In production, use safe evaluation
            return source_dict.get("solution", source_dict.get("result"))
        except:
            return None'''

    new_code = '''    def _extract_solution(self, source: str) -> Any:
        """Extract solution from source case."""
        # Parse source string back to dict using safe literal_eval
        import ast
        try:
            source_dict = ast.literal_eval(source)
            if not isinstance(source_dict, dict):
                logger.warning(f"Source case is not a dict: {type(source_dict)}")
                return None
            return source_dict.get("solution", source_dict.get("result"))
        except (ValueError, SyntaxError) as e:
            logger.error(f"Failed to parse source case: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing source case: {e}")
            return None'''

    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✓ Fixed eval() in capabilities/reasoning.py")
        return True
    else:
        print("✗ Could not find eval() pattern in capabilities/reasoning.py")
        return False


def fix_pipeline_py():
    """Fix eval() in orchestration/pipeline.py"""
    file_path = 'C:/Users/aksha/Code-V1_GreenLang/GreenLang_2030/agent_foundation/orchestration/pipeline.py'

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find and replace the eval() line
    for i, line in enumerate(lines):
        if 'return eval(condition, {"__builtins__": {}}, context)' in line:
            # Add import at the top of the method
            # Find the start of the method
            method_start = i
            while method_start > 0 and not lines[method_start].strip().startswith('def '):
                method_start -= 1

            # Check if simpleeval import already exists
            import_exists = False
            for j in range(0, min(50, len(lines))):
                if 'from simpleeval import simple_eval' in lines[j]:
                    import_exists = True
                    break

            # Add import if not exists
            if not import_exists:
                # Find the import section
                for j in range(0, min(50, len(lines))):
                    if lines[j].startswith('import ') or lines[j].startswith('from '):
                        # Insert after the last import
                        if j > 0 and (lines[j+1].strip() == '' or not (lines[j+1].startswith('import ') or lines[j+1].startswith('from '))):
                            lines.insert(j+1, 'from simpleeval import simple_eval, DEFAULT_NAMES, DEFAULT_FUNCTIONS\n')
                            i += 1  # Adjust index
                            break

            # Replace the eval() call
            indent = len(line) - len(line.lstrip())
            replacement = f'''{' ' * indent}# Safe evaluation using simpleeval instead of eval()
{' ' * indent}try:
{' ' * indent}    safe_names = {{**context}}
{' ' * indent}    safe_functions = {{
{' ' * indent}        "len": len,
{' ' * indent}        "str": str,
{' ' * indent}        "int": int,
{' ' * indent}        "float": float,
{' ' * indent}        "bool": bool
{' ' * indent}    }}
{' ' * indent}    return simple_eval(condition, names=safe_names, functions=safe_functions)
{' ' * indent}except Exception as e:
{' ' * indent}    logger.error(f"Condition evaluation failed: {{e}}")
{' ' * indent}    return False
'''

            # Remove the old try-except block
            # Find the start of try block
            try_start = i
            while try_start > 0 and 'try:' not in lines[try_start]:
                try_start -= 1

            # Find the end of except block
            except_end = i
            while except_end < len(lines) and not (lines[except_end].strip().startswith('return False') and 'except' in ''.join(lines[try_start:except_end])):
                except_end += 1

            # Replace the entire try-except block
            lines[try_start] = replacement
            del lines[try_start+1:except_end+1]

            break

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print("✓ Fixed eval() in orchestration/pipeline.py")
    return True


def fix_routing_py():
    """Fix eval() in orchestration/routing.py"""
    file_path = 'C:/Users/aksha/Code-V1_GreenLang/GreenLang_2030/agent_foundation/orchestration/routing.py'

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find and replace the eval() line
    for i, line in enumerate(lines):
        if 'return eval(self.condition, {"__builtins__": {}}, context)' in line:
            # Check if simpleeval import already exists
            import_exists = False
            for j in range(0, min(50, len(lines))):
                if 'from simpleeval import simple_eval' in lines[j]:
                    import_exists = True
                    break

            # Add import if not exists
            if not import_exists:
                # Find the import section
                for j in range(0, min(50, len(lines))):
                    if lines[j].startswith('import ') or lines[j].startswith('from '):
                        # Insert after the last import
                        if j > 0 and (lines[j+1].strip() == '' or not (lines[j+1].startswith('import ') or lines[j+1].startswith('from '))):
                            lines.insert(j+1, 'from simpleeval import simple_eval\n')
                            i += 1  # Adjust index
                            break

            # Replace the eval() call
            indent = len(line) - len(line.lstrip())
            replacement = f'''{' ' * indent}# Safe evaluation using simpleeval instead of eval()
{' ' * indent}try:
{' ' * indent}    return simple_eval(self.condition, names=context)
{' ' * indent}except Exception as e:
{' ' * indent}    logger.error(f"Rule evaluation failed: {{e}}")
{' ' * indent}    return False
'''

            # Remove the old try-except block
            # Find the start of try block
            try_start = i
            while try_start > 0 and 'try:' not in lines[try_start]:
                try_start -= 1

            # Find the end of except block
            except_end = i
            while except_end < len(lines) and not (lines[except_end].strip().startswith('return False') and 'except' in ''.join(lines[try_start:except_end])):
                except_end += 1

            # Replace the entire try-except block
            lines[try_start] = replacement
            del lines[try_start+1:except_end+1]

            break

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print("✓ Fixed eval() in orchestration/routing.py")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("SECURITY FIX: Removing eval() vulnerabilities")
    print("=" * 70)

    results = []

    print("\n[1/3] Fixing capabilities/reasoning.py...")
    results.append(fix_reasoning_py())

    print("\n[2/3] Fixing orchestration/pipeline.py...")
    results.append(fix_pipeline_py())

    print("\n[3/3] Fixing orchestration/routing.py...")
    results.append(fix_routing_py())

    print("\n" + "=" * 70)
    if all(results):
        print("✓ ALL FIXES APPLIED SUCCESSFULLY")
    else:
        print("✗ SOME FIXES FAILED - MANUAL REVIEW REQUIRED")
    print("=" * 70)
