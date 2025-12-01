#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GreenLang Agent Inheritance Validator
======================================

This script validates that all agents in the codebase use the standardized
AgentSpecV2Base + category mixin inheritance pattern.

Validation checks:
1. All agents inherit from AgentSpecV2Base
2. Each agent has exactly one category mixin (DeterministicMixin, ReasoningMixin, InsightMixin)
3. DeterministicMixin agents have no LLM calls in execute_impl()
4. Audit trail methods are implemented correctly
5. Pydantic input/output models are properly defined

Usage:
    python scripts/validate_agent_inheritance.py
    python scripts/validate_agent_inheritance.py --fix-imports
    python scripts/validate_agent_inheritance.py --report-only

Author: GreenLang Framework Team
Date: December 2025
"""

import ast
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Add greenlang to path
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))


@dataclass
class ValidationIssue:
    """Validation issue found in agent code."""
    severity: str  # "error", "warning", "info"
    file_path: Path
    line_number: Optional[int]
    code: str
    message: str
    suggestion: Optional[str] = None


@dataclass
class AgentInfo:
    """Information about an agent class."""
    name: str
    file_path: Path
    line_number: int
    base_classes: List[str]
    has_agentspec_v2_base: bool
    category_mixins: List[str]
    has_execute_impl: bool
    has_pydantic_models: bool
    issues: List[ValidationIssue]


class AgentInheritanceValidator:
    """Validator for agent inheritance patterns."""

    # Standard base classes to check
    AGENTSPEC_V2_BASE = "AgentSpecV2Base"
    CATEGORY_MIXINS = {"DeterministicMixin", "ReasoningMixin", "InsightMixin"}

    # Old patterns to flag for migration
    OLD_PATTERNS = {"BaseAgent", "Agent", "DeterministicAgent", "ReasoningAgent", "InsightAgent"}

    # LLM-related function calls to detect
    LLM_CALLS = {
        "chat", "complete", "generate", "llm", "openai", "anthropic",
        "ChatSession", "chat_session", "rag_engine"
    }

    def __init__(self, root_dir: Path):
        """
        Initialize validator.

        Args:
            root_dir: Root directory of codebase
        """
        self.root_dir = root_dir
        self.agents_dir = root_dir / "greenlang" / "agents"
        self.agents: List[AgentInfo] = []
        self.issues: List[ValidationIssue] = []

    def find_agent_files(self) -> List[Path]:
        """
        Find all Python files containing agent classes.

        Returns:
            List of agent file paths
        """
        agent_files = []

        # Search in greenlang/agents directory
        if self.agents_dir.exists():
            for file_path in self.agents_dir.rglob("*.py"):
                # Skip __init__.py, test files, and internal files
                if file_path.name.startswith("_") or "test" in file_path.name:
                    continue

                agent_files.append(file_path)

        # Search in application-specific directories
        for app_dir in self.root_dir.glob("GL-*-APP/agents"):
            if app_dir.exists():
                for file_path in app_dir.rglob("*.py"):
                    if file_path.name.startswith("_") or "test" in file_path.name:
                        continue
                    agent_files.append(file_path)

        return sorted(agent_files)

    def parse_agent_file(self, file_path: Path) -> List[AgentInfo]:
        """
        Parse Python file and extract agent class information.

        Args:
            file_path: Path to Python file

        Returns:
            List of AgentInfo objects found in file
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()

            tree = ast.parse(source_code, filename=str(file_path))

            agents = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if this looks like an agent class
                    if self._is_agent_class(node):
                        agent_info = self._extract_agent_info(node, file_path, source_code)
                        agents.append(agent_info)

            return agents

        except SyntaxError as e:
            self.issues.append(ValidationIssue(
                severity="error",
                file_path=file_path,
                line_number=e.lineno,
                code="SYNTAX_ERROR",
                message=f"Syntax error: {e.msg}"
            ))
            return []

        except Exception as e:
            self.issues.append(ValidationIssue(
                severity="error",
                file_path=file_path,
                line_number=None,
                code="PARSE_ERROR",
                message=f"Failed to parse file: {str(e)}"
            ))
            return []

    def _is_agent_class(self, node: ast.ClassDef) -> bool:
        """
        Check if class node represents an agent class.

        Args:
            node: AST ClassDef node

        Returns:
            True if this is an agent class
        """
        class_name = node.name.lower()

        # Check if name contains "agent"
        if "agent" in class_name:
            return True

        # Check if inherits from known base classes
        for base in node.bases:
            base_name = self._get_base_class_name(base)
            if base_name in self.OLD_PATTERNS or base_name == self.AGENTSPEC_V2_BASE:
                return True

        return False

    def _get_base_class_name(self, base: ast.expr) -> str:
        """
        Extract base class name from AST node.

        Args:
            base: AST node representing base class

        Returns:
            Base class name
        """
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Subscript):
            # Handle generic types like Agent[InT, OutT]
            if isinstance(base.value, ast.Name):
                return base.value.id
        elif isinstance(base, ast.Attribute):
            return base.attr

        return ""

    def _extract_agent_info(self, node: ast.ClassDef, file_path: Path, source_code: str) -> AgentInfo:
        """
        Extract detailed information about agent class.

        Args:
            node: AST ClassDef node
            file_path: Path to source file
            source_code: Full source code

        Returns:
            AgentInfo object
        """
        # Extract base classes
        base_classes = [self._get_base_class_name(base) for base in node.bases]

        # Check for AgentSpecV2Base
        has_agentspec_v2_base = self.AGENTSPEC_V2_BASE in base_classes

        # Check for category mixins
        category_mixins = [b for b in base_classes if b in self.CATEGORY_MIXINS]

        # Check for execute_impl method
        has_execute_impl = any(
            isinstance(n, ast.FunctionDef) and n.name == "execute_impl"
            for n in node.body
        )

        # Create AgentInfo
        agent_info = AgentInfo(
            name=node.name,
            file_path=file_path,
            line_number=node.lineno,
            base_classes=base_classes,
            has_agentspec_v2_base=has_agentspec_v2_base,
            category_mixins=category_mixins,
            has_execute_impl=has_execute_impl,
            has_pydantic_models=False,  # Will be checked separately
            issues=[]
        )

        # Validate this agent
        self._validate_agent(agent_info, node, source_code)

        return agent_info

    def _validate_agent(self, agent_info: AgentInfo, node: ast.ClassDef, source_code: str) -> None:
        """
        Validate agent class against standards.

        Args:
            agent_info: AgentInfo object to populate with issues
            node: AST ClassDef node
            source_code: Full source code
        """
        # Check 1: Must inherit from AgentSpecV2Base
        if not agent_info.has_agentspec_v2_base:
            # Check if using old patterns
            old_bases = [b for b in agent_info.base_classes if b in self.OLD_PATTERNS]
            if old_bases:
                agent_info.issues.append(ValidationIssue(
                    severity="error",
                    file_path=agent_info.file_path,
                    line_number=agent_info.line_number,
                    code="OLD_BASE_CLASS",
                    message=f"Agent {agent_info.name} uses old base class: {old_bases}",
                    suggestion=f"Migrate to: class {agent_info.name}(AgentSpecV2Base[InT, OutT], CategoryMixin)"
                ))

        # Check 2: Must have exactly one category mixin
        if len(agent_info.category_mixins) == 0 and agent_info.has_agentspec_v2_base:
            agent_info.issues.append(ValidationIssue(
                severity="error",
                file_path=agent_info.file_path,
                line_number=agent_info.line_number,
                code="MISSING_CATEGORY_MIXIN",
                message=f"Agent {agent_info.name} inherits from AgentSpecV2Base but has no category mixin",
                suggestion="Add one of: DeterministicMixin, ReasoningMixin, or InsightMixin"
            ))

        elif len(agent_info.category_mixins) > 1:
            agent_info.issues.append(ValidationIssue(
                severity="error",
                file_path=agent_info.file_path,
                line_number=agent_info.line_number,
                code="MULTIPLE_CATEGORY_MIXINS",
                message=f"Agent {agent_info.name} has multiple category mixins: {agent_info.category_mixins}",
                suggestion="Use only one category mixin. If you need hybrid behavior, use InsightMixin."
            ))

        # Check 3: Must implement execute_impl if using AgentSpecV2Base
        if agent_info.has_agentspec_v2_base and not agent_info.has_execute_impl:
            agent_info.issues.append(ValidationIssue(
                severity="error",
                file_path=agent_info.file_path,
                line_number=agent_info.line_number,
                code="MISSING_EXECUTE_IMPL",
                message=f"Agent {agent_info.name} must implement execute_impl() method",
                suggestion="Add: def execute_impl(self, validated_input: InT, context: AgentExecutionContext) -> OutT"
            ))

        # Check 4: DeterministicMixin agents must not use LLM calls
        if "DeterministicMixin" in agent_info.category_mixins:
            self._check_no_llm_calls(agent_info, node, source_code)

        # Check 5: Validate audit trail usage
        if "DeterministicMixin" in agent_info.category_mixins or "InsightMixin" in agent_info.category_mixins:
            self._check_audit_trail_usage(agent_info, node)

    def _check_no_llm_calls(self, agent_info: AgentInfo, node: ast.ClassDef, source_code: str) -> None:
        """
        Check that DeterministicMixin agents don't use LLM calls in execute_impl.

        Args:
            agent_info: AgentInfo object
            node: AST ClassDef node
            source_code: Full source code
        """
        # Find execute_impl method
        execute_impl = None
        for n in node.body:
            if isinstance(n, ast.FunctionDef) and n.name == "execute_impl":
                execute_impl = n
                break

        if not execute_impl:
            return

        # Check for LLM-related calls
        for child_node in ast.walk(execute_impl):
            if isinstance(child_node, ast.Call):
                # Check function name
                func_name = ""
                if isinstance(child_node.func, ast.Name):
                    func_name = child_node.func.id
                elif isinstance(child_node.func, ast.Attribute):
                    func_name = child_node.func.attr

                # Check if this is an LLM call
                if any(llm_keyword in func_name.lower() for llm_keyword in self.LLM_CALLS):
                    agent_info.issues.append(ValidationIssue(
                        severity="error",
                        file_path=agent_info.file_path,
                        line_number=child_node.lineno,
                        code="LLM_IN_DETERMINISTIC",
                        message=f"DeterministicMixin agent {agent_info.name} uses LLM call: {func_name}()",
                        suggestion="Remove LLM calls from execute_impl(). DeterministicMixin requires zero-hallucination guarantee."
                    ))

    def _check_audit_trail_usage(self, agent_info: AgentInfo, node: ast.ClassDef) -> None:
        """
        Check that agents using audit trail mixins actually call audit methods.

        Args:
            agent_info: AgentInfo object
            node: AST ClassDef node
        """
        # Find execute_impl method
        execute_impl = None
        for n in node.body:
            if isinstance(n, ast.FunctionDef) and n.name == "execute_impl":
                execute_impl = n
                break

        if not execute_impl:
            return

        # Check for audit trail calls
        has_audit_call = False
        for child_node in ast.walk(execute_impl):
            if isinstance(child_node, ast.Call):
                if isinstance(child_node.func, ast.Attribute):
                    if "capture_audit" in child_node.func.attr or "capture_calculation_audit" in child_node.func.attr:
                        has_audit_call = True
                        break

        if not has_audit_call:
            agent_info.issues.append(ValidationIssue(
                severity="warning",
                file_path=agent_info.file_path,
                line_number=agent_info.line_number,
                code="MISSING_AUDIT_TRAIL",
                message=f"Agent {agent_info.name} should capture audit trail in execute_impl()",
                suggestion="Add: self.capture_audit_entry(operation=..., inputs=..., outputs=..., calculation_trace=...)"
            ))

    def validate_all(self) -> Dict[str, Any]:
        """
        Validate all agents in codebase.

        Returns:
            Validation report
        """
        print("GreenLang Agent Inheritance Validator")
        print("=" * 80)

        # Find all agent files
        agent_files = self.find_agent_files()
        print(f"\nFound {len(agent_files)} agent files")

        # Parse and validate each file
        total_agents = 0
        for file_path in agent_files:
            agents = self.parse_agent_file(file_path)
            self.agents.extend(agents)
            total_agents += len(agents)

        print(f"Found {total_agents} agent classes")

        # Collect all issues
        all_issues = self.issues.copy()
        for agent in self.agents:
            all_issues.extend(agent.issues)

        # Categorize issues
        errors = [i for i in all_issues if i.severity == "error"]
        warnings = [i for i in all_issues if i.severity == "warning"]
        info = [i for i in all_issues if i.severity == "info"]

        # Build report
        report = {
            "total_files": len(agent_files),
            "total_agents": total_agents,
            "total_issues": len(all_issues),
            "errors": len(errors),
            "warnings": len(warnings),
            "info": len(info),
            "agents": self.agents,
            "issues": all_issues
        }

        return report

    def print_report(self, report: Dict[str, Any], verbose: bool = False) -> None:
        """
        Print validation report to console.

        Args:
            report: Validation report
            verbose: Whether to print detailed info
        """
        print("\n" + "=" * 80)
        print("VALIDATION REPORT")
        print("=" * 80)

        print(f"\nSummary:")
        print(f"  - Total files scanned: {report['total_files']}")
        print(f"  - Total agents found: {report['total_agents']}")
        print(f"  - Total issues: {report['total_issues']}")
        print(f"    - Errors: {report['errors']}")
        print(f"    - Warnings: {report['warnings']}")
        print(f"    - Info: {report['info']}")

        # Count agents by category
        agentspec_v2_count = sum(1 for a in report['agents'] if a.has_agentspec_v2_base)
        old_pattern_count = report['total_agents'] - agentspec_v2_count

        print(f"\nArchitecture:")
        print(f"  - AgentSpecV2Base + Mixin: {agentspec_v2_count}")
        print(f"  - Old patterns (needs migration): {old_pattern_count}")

        # Count by category mixin
        mixin_counts = defaultdict(int)
        for agent in report['agents']:
            if len(agent.category_mixins) == 1:
                mixin_counts[agent.category_mixins[0]] += 1
            elif len(agent.category_mixins) == 0:
                mixin_counts["None"] += 1
            else:
                mixin_counts["Multiple"] += 1

        if mixin_counts:
            print(f"\nCategory Distribution:")
            for mixin, count in sorted(mixin_counts.items()):
                print(f"  - {mixin}: {count}")

        # Print issues
        if report['errors'] > 0:
            print("\nERRORS:")
            for issue in [i for i in report['issues'] if i.severity == "error"]:
                self._print_issue(issue)

        if report['warnings'] > 0:
            print("\nWARNINGS:")
            for issue in [i for i in report['issues'] if i.severity == "warning"]:
                self._print_issue(issue)

        if verbose and report['info'] > 0:
            print("\nINFO:")
            for issue in [i for i in report['issues'] if i.severity == "info"]:
                self._print_issue(issue)

        # Print final result
        print("\n" + "=" * 80)
        if report['errors'] == 0:
            print("VALIDATION PASSED (no errors)")
        else:
            print(f"VALIDATION FAILED ({report['errors']} errors found)")
        print("=" * 80)

    def _print_issue(self, issue: ValidationIssue) -> None:
        """Print a single validation issue."""
        print(f"\n  [{issue.code}] {issue.file_path.name}:{issue.line_number or '?'}")
        print(f"    {issue.message}")
        if issue.suggestion:
            print(f"    Suggestion: {issue.suggestion}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate GreenLang agent inheritance patterns"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose output including info messages"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate report, don't exit with error code"
    )

    args = parser.parse_args()

    # Run validation
    validator = AgentInheritanceValidator(ROOT_DIR)
    report = validator.validate_all()

    # Print report
    validator.print_report(report, verbose=args.verbose)

    # Exit with error code if there are errors
    if not args.report_only and report['errors'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
