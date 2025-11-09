#!/usr/bin/env python3
"""
Auto-Documentation Generator

Generate API documentation from code.
Extracts docstrings, type hints, and examples.
"""

import argparse
import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
import json
import re


@dataclass
class FunctionDoc:
    """Function documentation."""
    name: str
    signature: str
    docstring: str
    parameters: List[Dict[str, str]]
    returns: str
    examples: List[str]


@dataclass
class ClassDoc:
    """Class documentation."""
    name: str
    docstring: str
    methods: List[FunctionDoc]
    attributes: List[Dict[str, str]]
    base_classes: List[str]


@dataclass
class ModuleDoc:
    """Module documentation."""
    name: str
    file_path: str
    docstring: str
    classes: List[ClassDoc]
    functions: List[FunctionDoc]


class DocExtractor:
    """Extract documentation from Python code."""

    def extract_module(self, file_path: str) -> ModuleDoc:
        """Extract documentation from a Python module."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return None

        module_name = Path(file_path).stem
        module_docstring = ast.get_docstring(tree) or ""

        classes = []
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_doc = self._extract_class(node)
                if class_doc:
                    classes.append(class_doc)
            elif isinstance(node, ast.FunctionDef):
                # Only top-level functions
                if node.col_offset == 0:
                    func_doc = self._extract_function(node)
                    if func_doc:
                        functions.append(func_doc)

        return ModuleDoc(
            name=module_name,
            file_path=file_path,
            docstring=module_docstring,
            classes=classes,
            functions=functions
        )

    def _extract_class(self, node: ast.ClassDef) -> ClassDoc:
        """Extract class documentation."""
        docstring = ast.get_docstring(node) or ""

        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_doc = self._extract_function(item)
                if method_doc:
                    methods.append(method_doc)

        base_classes = [base.id for base in node.bases if hasattr(base, 'id')]

        return ClassDoc(
            name=node.name,
            docstring=docstring,
            methods=methods,
            attributes=[],
            base_classes=base_classes
        )

    def _extract_function(self, node: ast.FunctionDef) -> FunctionDoc:
        """Extract function documentation."""
        docstring = ast.get_docstring(node) or ""

        # Extract parameters
        parameters = []
        for arg in node.args.args:
            param = {
                "name": arg.arg,
                "type": self._get_annotation(arg.annotation),
                "description": ""
            }
            parameters.append(param)

        # Extract return type
        returns = self._get_annotation(node.returns)

        # Build signature
        param_strs = []
        for arg in node.args.args:
            param_str = arg.arg
            if hasattr(arg, 'annotation') and arg.annotation:
                param_str += f": {self._get_annotation(arg.annotation)}"
            param_strs.append(param_str)

        signature = f"{node.name}({', '.join(param_strs)})"
        if returns:
            signature += f" -> {returns}"

        # Extract examples from docstring
        examples = self._extract_examples(docstring)

        return FunctionDoc(
            name=node.name,
            signature=signature,
            docstring=docstring,
            parameters=parameters,
            returns=returns,
            examples=examples
        )

    def _get_annotation(self, annotation) -> str:
        """Get type annotation as string."""
        if annotation is None:
            return ""

        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Subscript):
            # e.g., List[str], Dict[str, Any]
            return ast.unparse(annotation)
        else:
            return ast.unparse(annotation) if hasattr(ast, 'unparse') else ""

    def _extract_examples(self, docstring: str) -> List[str]:
        """Extract code examples from docstring."""
        examples = []

        # Look for code blocks
        code_block_pattern = r'```python\n(.*?)\n```'
        matches = re.findall(code_block_pattern, docstring, re.DOTALL)
        examples.extend(matches)

        # Look for Example: sections
        example_pattern = r'Example:\s*\n((?:    .*\n)*)'
        matches = re.findall(example_pattern, docstring)
        examples.extend([m.strip() for m in matches if m.strip()])

        return examples


class MarkdownGenerator:
    """Generate Markdown documentation."""

    def generate(self, modules: List[ModuleDoc], output_dir: str):
        """Generate Markdown docs."""
        os.makedirs(output_dir, exist_ok=True)

        # Index page
        self._generate_index(modules, output_dir)

        # Module pages
        for module in modules:
            self._generate_module_page(module, output_dir)

    def _generate_index(self, modules: List[ModuleDoc], output_dir: str):
        """Generate index page."""
        md = ["# Infrastructure API Reference\n"]
        md.append("## Modules\n")

        for module in modules:
            md.append(f"- [{module.name}]({module.name}.md)")
            if module.docstring:
                first_line = module.docstring.split('\n')[0]
                md.append(f"  - {first_line}")
            md.append("")

        with open(os.path.join(output_dir, 'index.md'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(md))

    def _generate_module_page(self, module: ModuleDoc, output_dir: str):
        """Generate module documentation page."""
        md = [f"# {module.name}\n"]

        if module.docstring:
            md.append(module.docstring)
            md.append("")

        # Classes
        if module.classes:
            md.append("## Classes\n")
            for cls in module.classes:
                md.extend(self._format_class(cls))

        # Functions
        if module.functions:
            md.append("## Functions\n")
            for func in module.functions:
                md.extend(self._format_function(func))

        with open(os.path.join(output_dir, f'{module.name}.md'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(md))

    def _format_class(self, cls: ClassDoc) -> List[str]:
        """Format class documentation."""
        md = [f"### {cls.name}\n"]

        if cls.base_classes:
            md.append(f"**Inherits from:** {', '.join(cls.base_classes)}\n")

        if cls.docstring:
            md.append(cls.docstring)
            md.append("")

        # Methods
        if cls.methods:
            md.append("**Methods:**\n")
            for method in cls.methods:
                md.append(f"#### {method.name}\n")
                md.append(f"```python\n{method.signature}\n```\n")

                if method.docstring:
                    md.append(method.docstring)
                    md.append("")

                if method.examples:
                    md.append("**Example:**\n")
                    for example in method.examples:
                        md.append(f"```python\n{example}\n```\n")

        md.append("---\n")
        return md

    def _format_function(self, func: FunctionDoc) -> List[str]:
        """Format function documentation."""
        md = [f"### {func.name}\n"]
        md.append(f"```python\n{func.signature}\n```\n")

        if func.docstring:
            md.append(func.docstring)
            md.append("")

        if func.parameters:
            md.append("**Parameters:**\n")
            for param in func.parameters:
                type_str = f" ({param['type']})" if param['type'] else ""
                md.append(f"- `{param['name']}`{type_str}")
            md.append("")

        if func.returns:
            md.append(f"**Returns:** {func.returns}\n")

        if func.examples:
            md.append("**Example:**\n")
            for example in func.examples:
                md.append(f"```python\n{example}\n```\n")

        md.append("---\n")
        return md


class HTMLGenerator:
    """Generate HTML documentation."""

    def generate(self, modules: List[ModuleDoc], output_dir: str):
        """Generate HTML docs."""
        os.makedirs(output_dir, exist_ok=True)

        # Generate pages
        self._generate_index(modules, output_dir)

        for module in modules:
            self._generate_module_page(module, output_dir)

    def _generate_index(self, modules: List[ModuleDoc], output_dir: str):
        """Generate HTML index."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Infrastructure API Reference</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; }
        .module-list { list-style: none; padding: 0; }
        .module-list li { padding: 10px; margin: 5px 0; background: #ecf0f1; border-radius: 5px; }
        .module-list a { color: #2c3e50; text-decoration: none; font-weight: bold; }
        .module-list a:hover { color: #3498db; }
    </style>
</head>
<body>
    <h1>Infrastructure API Reference</h1>
    <ul class="module-list">
"""

        for module in modules:
            html += f'<li><a href="{module.name}.html">{module.name}</a>'
            if module.docstring:
                first_line = module.docstring.split('\n')[0]
                html += f'<br><small>{first_line}</small>'
            html += '</li>\n'

        html += """
    </ul>
</body>
</html>
"""

        with open(os.path.join(output_dir, 'index.html'), 'w', encoding='utf-8') as f:
            f.write(html)

    def _generate_module_page(self, module: ModuleDoc, output_dir: str):
        """Generate module HTML page."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{module.name} - API Reference</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
        .class, .function {{ background: #f9f9f9; padding: 20px; margin: 20px 0; border-left: 4px solid #3498db; }}
        code {{ background: #2c3e50; color: #ecf0f1; padding: 2px 5px; border-radius: 3px; }}
        pre {{ background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>{module.name}</h1>
    <p>{module.docstring}</p>
"""

        # Classes
        if module.classes:
            html += "<h2>Classes</h2>\n"
            for cls in module.classes:
                html += f'<div class="class">\n<h3>{cls.name}</h3>\n'
                html += f'<p>{cls.docstring}</p>\n'

                for method in cls.methods:
                    html += f'<h4>{method.name}</h4>\n'
                    html += f'<pre>{method.signature}</pre>\n'
                    html += f'<p>{method.docstring}</p>\n'

                html += '</div>\n'

        # Functions
        if module.functions:
            html += "<h2>Functions</h2>\n"
            for func in module.functions:
                html += f'<div class="function">\n<h3>{func.name}</h3>\n'
                html += f'<pre>{func.signature}</pre>\n'
                html += f'<p>{func.docstring}</p>\n'
                html += '</div>\n'

        html += """
</body>
</html>
"""

        with open(os.path.join(output_dir, f'{module.name}.html'), 'w', encoding='utf-8') as f:
            f.write(html)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Auto-generate documentation')
    parser.add_argument('--source', default='shared/infrastructure', help='Source directory')
    parser.add_argument('--output', default='docs', help='Output directory')
    parser.add_argument('--format', choices=['markdown', 'html', 'both'], default='both')

    args = parser.parse_args()

    print(f"Generating documentation from {args.source}...")

    # Extract documentation
    extractor = DocExtractor()
    modules = []

    if os.path.exists(args.source):
        for root, dirs, files in os.walk(args.source):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    file_path = os.path.join(root, file)
                    module_doc = extractor.extract_module(file_path)
                    if module_doc:
                        modules.append(module_doc)

    print(f"Found {len(modules)} modules")

    # Generate documentation
    if args.format in ['markdown', 'both']:
        print("Generating Markdown documentation...")
        md_gen = MarkdownGenerator()
        md_gen.generate(modules, os.path.join(args.output, 'markdown'))
        print(f"Markdown docs: {os.path.join(args.output, 'markdown')}")

    if args.format in ['html', 'both']:
        print("Generating HTML documentation...")
        html_gen = HTMLGenerator()
        html_gen.generate(modules, os.path.join(args.output, 'html'))
        print(f"HTML docs: {os.path.join(args.output, 'html')}")

    print("\nDocumentation generated successfully!")


if __name__ == '__main__':
    main()
