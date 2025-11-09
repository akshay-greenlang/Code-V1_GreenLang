#!/usr/bin/env python3
"""
GreenLang Import Rewriter

AST-based tool to rewrite imports from external libraries to GreenLang equivalents.
Preserves code formatting and generates git diffs.
"""

import ast
import astor
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
import subprocess


@dataclass
class ImportMapping:
    """Mapping from old import to new import."""
    old_module: str
    old_name: str
    new_module: str
    new_name: str
    rewrite_references: bool = True


class ImportRewriter(ast.NodeTransformer):
    """AST transformer that rewrites imports."""

    def __init__(self, mappings: List[ImportMapping]):
        self.mappings = mappings
        self.rewritten = []
        self.reference_map = {}  # Map old names to new names in current file

        # Build lookup tables
        self.module_map = {}
        self.name_map = {}

        for mapping in mappings:
            key = f"{mapping.old_module}.{mapping.old_name}"
            self.module_map[mapping.old_module] = mapping
            self.name_map[key] = mapping

    def visit_Import(self, node: ast.Import) -> ast.AST:
        """Rewrite 'import X' statements."""
        new_names = []
        modified = False

        for alias in node.names:
            # Check if this import should be rewritten
            if alias.name in self.module_map:
                mapping = self.module_map[alias.name]
                new_name = ast.alias(
                    name=mapping.new_module,
                    asname=alias.asname
                )
                new_names.append(new_name)
                modified = True

                # Track name mapping for reference rewriting
                old_ref = alias.asname or alias.name
                new_ref = alias.asname or mapping.new_module
                self.reference_map[old_ref] = new_ref

                self.rewritten.append({
                    'type': 'import',
                    'old': f"import {alias.name}",
                    'new': f"import {mapping.new_module}"
                })
            else:
                new_names.append(alias)

        if modified:
            return ast.Import(names=new_names)
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        """Rewrite 'from X import Y' statements."""
        if node.module is None:
            return node

        # Check if entire module should be rewritten
        if node.module in self.module_map:
            mapping = self.module_map[node.module]
            new_node = ast.ImportFrom(
                module=mapping.new_module,
                names=node.names,
                level=node.level
            )

            self.rewritten.append({
                'type': 'import_from',
                'old': f"from {node.module} import ...",
                'new': f"from {mapping.new_module} import ..."
            })

            return new_node

        # Check if specific imports should be rewritten
        new_names = []
        modified = False

        for alias in node.names:
            key = f"{node.module}.{alias.name}"
            if key in self.name_map:
                mapping = self.name_map[key]

                # Track name mapping
                old_ref = alias.asname or alias.name
                new_ref = alias.asname or mapping.new_name
                self.reference_map[old_ref] = new_ref

                # If the module changed, we need to create a new import
                if mapping.new_module != node.module:
                    # Create separate import statement
                    # This will be handled separately
                    modified = True
                    self.rewritten.append({
                        'type': 'import_from',
                        'old': f"from {node.module} import {alias.name}",
                        'new': f"from {mapping.new_module} import {mapping.new_name}"
                    })
                else:
                    new_alias = ast.alias(
                        name=mapping.new_name,
                        asname=alias.asname
                    )
                    new_names.append(new_alias)
                    modified = True
            else:
                new_names.append(alias)

        if modified and new_names:
            return ast.ImportFrom(
                module=node.module,
                names=new_names,
                level=node.level
            )

        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        """Rewrite name references."""
        if node.id in self.reference_map:
            new_node = ast.Name(
                id=self.reference_map[node.id],
                ctx=node.ctx
            )
            return new_node
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        """Rewrite attribute references."""
        # Handle cases like 'openai.ChatCompletion'
        if isinstance(node.value, ast.Name):
            if node.value.id in self.reference_map:
                new_value = ast.Name(
                    id=self.reference_map[node.value.id],
                    ctx=node.value.ctx
                )
                return ast.Attribute(
                    value=new_value,
                    attr=node.attr,
                    ctx=node.ctx
                )
        return node


class ImportRewriterTool:
    """Main import rewriter tool."""

    def __init__(self):
        self.mappings = self._initialize_mappings()
        self.stats = {
            'files_processed': 0,
            'files_modified': 0,
            'imports_rewritten': 0
        }

    def _initialize_mappings(self) -> List[ImportMapping]:
        """Initialize import mappings."""
        return [
            # OpenAI
            ImportMapping(
                old_module="openai",
                old_name="OpenAI",
                new_module="greenlang.intelligence",
                new_name="ChatSession"
            ),
            ImportMapping(
                old_module="openai",
                old_name="ChatCompletion",
                new_module="greenlang.intelligence",
                new_name="ChatSession"
            ),

            # Anthropic
            ImportMapping(
                old_module="anthropic",
                old_name="Anthropic",
                new_module="greenlang.intelligence",
                new_name="ChatSession"
            ),

            # Redis
            ImportMapping(
                old_module="redis",
                old_name="Redis",
                new_module="greenlang.cache",
                new_name="CacheManager"
            ),

            # JSONSchema
            ImportMapping(
                old_module="jsonschema",
                old_name="validate",
                new_module="greenlang.validation",
                new_name="ValidationFramework"
            ),

            # LangChain
            ImportMapping(
                old_module="langchain",
                old_name="Chain",
                new_module="greenlang.sdk.base",
                new_name="Pipeline"
            ),
            ImportMapping(
                old_module="langchain",
                old_name="LLMChain",
                new_module="greenlang.sdk.base",
                new_name="Pipeline"
            ),

            # Requests
            ImportMapping(
                old_module="requests",
                old_name="Session",
                new_module="greenlang.utils.http",
                new_name="HTTPClient"
            ),

            # Celery (if using for background tasks)
            ImportMapping(
                old_module="celery",
                old_name="Celery",
                new_module="greenlang.tasks",
                new_name="TaskQueue"
            ),

            # Pydantic (use GreenLang models)
            ImportMapping(
                old_module="pydantic",
                old_name="BaseModel",
                new_module="greenlang.models",
                new_name="BaseModel"
            ),
        ]

    def rewrite_file(self, file_path: str, dry_run: bool = True) -> Tuple[bool, str, str]:
        """
        Rewrite imports in a file.

        Returns:
            (modified, original_content, new_content)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            # Parse AST
            try:
                tree = ast.parse(original_content)
            except SyntaxError as e:
                print(f"Syntax error in {file_path}: {e}")
                return False, original_content, original_content

            # Apply transformations
            rewriter = ImportRewriter(self.mappings)
            new_tree = rewriter.visit(tree)
            ast.fix_missing_locations(new_tree)

            # Generate new code
            try:
                new_content = astor.to_source(new_tree)
            except Exception as e:
                print(f"Error generating code for {file_path}: {e}")
                return False, original_content, original_content

            # Check if modified
            modified = len(rewriter.rewritten) > 0

            if modified:
                self.stats['imports_rewritten'] += len(rewriter.rewritten)

                if not dry_run:
                    # Write back
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    self.stats['files_modified'] += 1

            self.stats['files_processed'] += 1

            return modified, original_content, new_content

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False, "", ""

    def rewrite_directory(self, directory: str, dry_run: bool = True, exclude_patterns: List[str] = None):
        """Rewrite all Python files in a directory."""
        if exclude_patterns is None:
            exclude_patterns = ['__pycache__', '.git', 'venv', 'env', 'node_modules', '.greenlang']

        modified_files = []

        for root, dirs, files in os.walk(directory):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    modified, original, new = self.rewrite_file(file_path, dry_run=dry_run)

                    if modified:
                        modified_files.append({
                            'path': file_path,
                            'original': original,
                            'new': new
                        })

        return modified_files

    def generate_diff(self, original: str, new: str, file_path: str) -> str:
        """Generate unified diff."""
        import difflib

        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm=''
        )

        return ''.join(diff)

    def generate_git_diff(self, directory: str) -> str:
        """Generate git diff for all changes."""
        try:
            result = subprocess.run(
                ['git', 'diff'],
                cwd=directory,
                capture_output=True,
                text=True
            )
            return result.stdout
        except Exception as e:
            print(f"Error generating git diff: {e}")
            return ""


def main():
    parser = argparse.ArgumentParser(
        description="GreenLang Import Rewriter - Rewrite imports to use GreenLang infrastructure"
    )

    parser.add_argument(
        'path',
        help='File or directory to rewrite'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without applying them'
    )

    parser.add_argument(
        '--show-diff',
        action='store_true',
        help='Show unified diff of changes'
    )

    parser.add_argument(
        '--git-diff',
        action='store_true',
        help='Generate git diff after changes'
    )

    args = parser.parse_args()

    # Initialize tool
    tool = ImportRewriterTool()

    # Process
    print(f"Processing {args.path}...")

    if os.path.isfile(args.path):
        modified, original, new = tool.rewrite_file(args.path, dry_run=args.dry_run)

        if modified:
            print(f"\n✓ Modified: {args.path}")

            if args.show_diff:
                print("\nDiff:")
                print(tool.generate_diff(original, new, args.path))

    elif os.path.isdir(args.path):
        modified_files = tool.rewrite_directory(args.path, dry_run=args.dry_run)

        print(f"\nProcessed {tool.stats['files_processed']} files")
        print(f"Modified {len(modified_files)} files")
        print(f"Rewrote {tool.stats['imports_rewritten']} imports")

        if modified_files:
            print("\nModified files:")
            for file_info in modified_files:
                print(f"  - {file_info['path']}")

                if args.show_diff:
                    print("\nDiff:")
                    print(tool.generate_diff(file_info['original'], file_info['new'], file_info['path']))
                    print()

        if args.git_diff and not args.dry_run:
            print("\nGit diff:")
            print(tool.generate_git_diff(args.path))

    else:
        print(f"Error: {args.path} is not a valid file or directory")
        sys.exit(1)

    if args.dry_run:
        print("\n(Dry run - no changes applied)")
    else:
        print("\n✓ Import rewriting complete!")


if __name__ == '__main__':
    main()
