#!/usr/bin/env python3
# -*- coding: utf-8 -*-

logger = logging.getLogger(__name__)
"""
GreenLang Agent Inheritance Converter

Converts custom agent classes to inherit from greenlang.integration.sdk.base.Agent.
Maps custom methods to Agent lifecycle hooks.
"""

import logging
import ast
import astor
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass


@dataclass
class AgentClass:
    """Represents an agent class found in code."""
    name: str
    base_classes: List[str]
    methods: List[str]
    file_path: str
    line_number: int


class AgentConverter(ast.NodeTransformer):
    """AST transformer that converts agent classes."""

    def __init__(self):
        self.converted_classes = []
        self.imports_to_add = set()

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        """Visit class definitions and convert agent classes."""

        # Check if this looks like an agent class
        if self._is_agent_class(node):
            # Convert the class
            new_node = self._convert_agent_class(node)
            self.converted_classes.append(node.name)
            self.imports_to_add.add("from greenlang.integration.sdk.base import Agent")
            return new_node

        return node

    def _is_agent_class(self, node: ast.ClassDef) -> bool:
        """Determine if a class should be converted to Agent."""

        # Check name patterns
        name_lower = node.name.lower()
        if 'agent' in name_lower:
            return True

        # Check if it has methods that look like agent methods
        agent_method_names = {
            'process', 'execute', 'run', 'handle',
            'process_batch', 'process_item', 'process_message',
            'analyze', 'generate', 'transform'
        }

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if item.name in agent_method_names:
                    return True

        return False

    def _convert_agent_class(self, node: ast.ClassDef) -> ast.ClassDef:
        """Convert a class to inherit from Agent."""

        # Update base classes
        new_bases = [ast.Name(id='Agent', ctx=ast.Load())]

        # Keep other bases if they're not object or similar
        for base in node.bases:
            if isinstance(base, ast.Name):
                if base.id not in ['object', 'ABC']:
                    new_bases.append(base)

        # Convert methods
        new_body = []
        has_execute = False
        has_init = False

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_name = item.name

                # Check for __init__
                if method_name == '__init__':
                    has_init = True
                    new_body.append(self._convert_init_method(item))

                # Convert common patterns to execute()
                elif method_name in ['process', 'run', 'handle', 'process_item']:
                    if not has_execute:
                        new_method = self._convert_to_execute(item)
                        new_body.append(new_method)
                        has_execute = True
                    else:
                        # Rename to avoid conflicts
                        item.name = f"_{method_name}"
                        new_body.append(item)

                # Convert batch processing
                elif method_name in ['process_batch', 'batch_process']:
                    # Add comment about using Agent.batch_process
                    comment = ast.Expr(value=ast.Constant(
                        value="Note: Consider using self.batch_process() from Agent base class"
                    ))
                    new_body.append(comment)
                    new_body.append(item)

                else:
                    new_body.append(item)
            else:
                new_body.append(item)

        # Add execute() if not present
        if not has_execute:
            new_body.insert(1 if has_init else 0, self._create_execute_method())

        # Create new class
        new_class = ast.ClassDef(
            name=node.name,
            bases=new_bases,
            keywords=node.keywords,
            body=new_body,
            decorator_list=node.decorator_list
        )

        return new_class

    def _convert_init_method(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Convert __init__ to call super().__init__()."""

        # Check if super().__init__() is already called
        has_super_call = False
        for item in node.body:
            if isinstance(item, ast.Expr):
                if isinstance(item.value, ast.Call):
                    if isinstance(item.value.func, ast.Attribute):
                        if item.value.func.attr == '__init__':
                            has_super_call = True
                            break

        # Add super().__init__() if not present
        if not has_super_call:
            super_call = ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Call(
                            func=ast.Name(id='super', ctx=ast.Load()),
                            args=[],
                            keywords=[]
                        ),
                        attr='__init__',
                        ctx=ast.Load()
                    ),
                    args=[],
                    keywords=[]
                )
            )
            node.body.insert(0, super_call)

        return node

    def _convert_to_execute(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Convert a method to execute()."""

        # Rename to execute
        node.name = 'execute'

        # Ensure parameter is input_data
        if len(node.args.args) > 1:  # self + at least one param
            node.args.args[1].arg = 'input_data'

        return node

    def _create_execute_method(self) -> ast.FunctionDef:
        """Create a default execute() method."""

        return ast.FunctionDef(
            name='execute',
            args=ast.arguments(
                posonlyargs=[],
                args=[
                    ast.arg(arg='self', annotation=None),
                    ast.arg(arg='input_data', annotation=None)
                ],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[]
            ),
            body=[
                ast.Expr(value=ast.Constant(
                    value="Execute agent logic. Override this method with your implementation."
                )),
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id='NotImplementedError', ctx=ast.Load()),
                        args=[ast.Constant(value="Subclass must implement execute()")],
                        keywords=[]
                    ),
                    cause=None
                )
            ],
            decorator_list=[],
            returns=None
        )

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Process module and add necessary imports."""

        # Visit all nodes first
        new_node = self.generic_visit(node)

        # Add imports if any classes were converted
        if self.imports_to_add:
            # Find existing imports
            import_index = 0
            for i, item in enumerate(new_node.body):
                if isinstance(item, (ast.Import, ast.ImportFrom)):
                    import_index = i + 1

            # Add new imports
            for import_str in self.imports_to_add:
                # Parse the import string
                import_ast = ast.parse(import_str).body[0]
                new_node.body.insert(import_index, import_ast)
                import_index += 1

        return new_node


class AgentConverterTool:
    """Main agent converter tool."""

    def __init__(self):
        self.stats = {
            'files_processed': 0,
            'files_modified': 0,
            'classes_converted': 0
        }

    def convert_file(self, file_path: str, dry_run: bool = True) -> tuple:
        """
        Convert agent classes in a file.

        Returns:
            (modified, original_content, new_content, converted_classes)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            # Parse AST
            try:
                tree = ast.parse(original_content)
            except SyntaxError as e:
                print(f"Syntax error in {file_path}: {e}")
                return False, original_content, original_content, []

            # Apply transformations
            converter = AgentConverter()
            new_tree = converter.visit(tree)
            ast.fix_missing_locations(new_tree)

            # Generate new code
            try:
                new_content = astor.to_source(new_tree)
            except Exception as e:
                print(f"Error generating code for {file_path}: {e}")
                return False, original_content, original_content, []

            # Check if modified
            modified = len(converter.converted_classes) > 0

            if modified:
                self.stats['classes_converted'] += len(converter.converted_classes)

                if not dry_run:
                    # Write back
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    self.stats['files_modified'] += 1

            self.stats['files_processed'] += 1

            return modified, original_content, new_content, converter.converted_classes

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False, "", "", []

    def convert_directory(self, directory: str, dry_run: bool = True, exclude_patterns: List[str] = None):
        """Convert all Python files in a directory."""
        if exclude_patterns is None:
            exclude_patterns = ['__pycache__', '.git', 'venv', 'env', 'node_modules', '.greenlang']

        converted_files = []

        for root, dirs, files in os.walk(directory):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    modified, original, new, classes = self.convert_file(file_path, dry_run=dry_run)

                    if modified:
                        converted_files.append({
                            'path': file_path,
                            'original': original,
                            'new': new,
                            'classes': classes
                        })

        return converted_files

    def generate_report(self, converted_files: List[Dict]) -> str:
        """Generate conversion report."""
        lines = []
        lines.append("=" * 80)
        lines.append("GreenLang Agent Conversion Report")
        lines.append("=" * 80)
        lines.append("")

        # Summary
        lines.append("SUMMARY:")
        lines.append(f"  Files processed: {self.stats['files_processed']}")
        lines.append(f"  Files modified: {len(converted_files)}")
        lines.append(f"  Classes converted: {self.stats['classes_converted']}")
        lines.append("")

        # Details
        if converted_files:
            lines.append("CONVERTED FILES:")
            lines.append("-" * 80)

            for file_info in converted_files:
                lines.append(f"\nFile: {file_info['path']}")
                lines.append(f"Converted classes: {', '.join(file_info['classes'])}")
                lines.append("")

        return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="GreenLang Agent Converter - Convert custom agents to inherit from Agent base class"
    )

    parser.add_argument(
        'path',
        help='File or directory to convert'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without applying them'
    )

    parser.add_argument(
        '--show-diff',
        action='store_true',
        help='Show code differences'
    )

    args = parser.parse_args()

    # Initialize tool
    tool = AgentConverterTool()

    # Process
    print(f"Converting agents in {args.path}...")

    if os.path.isfile(args.path):
        modified, original, new, classes = tool.convert_file(args.path, dry_run=args.dry_run)

        if modified:
            print(f"\n✓ Converted {len(classes)} class(es) in {args.path}")
            print(f"  Classes: {', '.join(classes)}")

            if args.show_diff:
                import difflib
                diff = difflib.unified_diff(
                    original.splitlines(keepends=True),
                    new.splitlines(keepends=True),
                    fromfile=args.path,
                    tofile=args.path
                )
                print("\nDiff:")
                print(''.join(diff))

    elif os.path.isdir(args.path):
        converted_files = tool.convert_directory(args.path, dry_run=args.dry_run)

        print(f"\n{tool.generate_report(converted_files)}")

        if args.show_diff and converted_files:
            import difflib
            for file_info in converted_files:
                print(f"\nDiff for {file_info['path']}:")
                diff = difflib.unified_diff(
                    file_info['original'].splitlines(keepends=True),
                    file_info['new'].splitlines(keepends=True),
                    fromfile=file_info['path'],
                    tofile=file_info['path']
                )
                print(''.join(diff))

    else:
        logger.error(f"{args.path} is not a valid file or directory")
        sys.exit(1)

    if args.dry_run:
        print("\n(Dry run - no changes applied)")
    else:
        print("\n✓ Agent conversion complete!")


if __name__ == '__main__':
    main()
