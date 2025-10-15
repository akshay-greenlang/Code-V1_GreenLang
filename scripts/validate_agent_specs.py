#!/usr/bin/env python3
"""
GreenLang AgentSpec V2.0 Validator
Validates agent specifications against template and enforces tool-first design.

Usage:
    python validate_agent_specs.py <path_to_spec.yaml>
    python validate_agent_specs.py --batch specs/
    python validate_agent_specs.py --all

Author: GreenLang AI Team
Version: 1.0.0
Created: 2025-10-13
"""

import yaml
import json
import sys
import io
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import re

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


@dataclass
class ValidationError:
    """Represents a validation error with context."""
    severity: str  # "error", "warning", "info"
    section: str
    field: str
    message: str
    line_number: Optional[int] = None
    expected: Optional[Any] = None
    actual: Optional[Any] = None

    def __str__(self) -> str:
        line_info = f" (line {self.line_number})" if self.line_number else ""
        severity_prefix = {
            "error": "❌ ERROR",
            "warning": "⚠️  WARNING",
            "info": "ℹ️  INFO"
        }
        prefix = severity_prefix.get(self.severity, "")

        msg = f"{prefix}{line_info}: [{self.section}] {self.field}\n   {self.message}"
        if self.expected is not None:
            msg += f"\n   Expected: {self.expected}"
        if self.actual is not None:
            msg += f"\n   Actual: {self.actual}"
        return msg


@dataclass
class ValidationReport:
    """Complete validation report for an agent spec."""
    file_path: str
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    info: List[ValidationError] = field(default_factory=list)

    def add_error(self, section: str, field: str, message: str, **kwargs):
        self.errors.append(ValidationError("error", section, field, message, **kwargs))
        self.is_valid = False

    def add_warning(self, section: str, field: str, message: str, **kwargs):
        self.warnings.append(ValidationError("warning", section, field, message, **kwargs))

    def add_info(self, section: str, field: str, message: str, **kwargs):
        self.info.append(ValidationError("info", section, field, message, **kwargs))

    def get_summary(self) -> str:
        """Generate summary report."""
        status = "✅ PASS" if self.is_valid else "❌ FAIL"
        lines = [
            "=" * 80,
            f"Validation Report: {Path(self.file_path).name}",
            "=" * 80,
            f"Status: {status}",
            f"Errors: {len(self.errors)}",
            f"Warnings: {len(self.warnings)}",
            f"Info: {len(self.info)}",
            "-" * 80,
        ]

        if self.errors:
            lines.append("\n🔴 ERRORS:")
            for error in self.errors:
                lines.append(str(error))
                lines.append("")

        if self.warnings:
            lines.append("\n🟡 WARNINGS:")
            for warning in self.warnings:
                lines.append(str(warning))
                lines.append("")

        if self.info:
            lines.append("\nℹ️  INFORMATION:")
            for info_item in self.info:
                lines.append(str(info_item))
                lines.append("")

        if self.is_valid:
            lines.append("\n✨ All validation checks passed!")
        else:
            lines.append(f"\n⚠️  Validation failed with {len(self.errors)} error(s)")

        lines.append("=" * 80)
        return "\n".join(lines)


class AgentSpecValidator:
    """Validates GreenLang AgentSpec V2.0 files."""

    # Required top-level sections
    REQUIRED_SECTIONS = [
        "agent_metadata",
        "description",
        "tools",
        "ai_integration",
        "inputs",
        "outputs",
        "testing",
        "deployment",
        "documentation",
        "compliance",
        "metadata"
    ]

    # Required fields in agent_metadata
    REQUIRED_AGENT_METADATA = [
        "agent_id",
        "agent_name",
        "version",
        "domain",
        "category",
        "complexity",
        "priority",
        "base_agent",
        "status"
    ]

    # Required fields in description
    REQUIRED_DESCRIPTION = [
        "purpose",
        "strategic_context",
        "key_capabilities"
    ]

    # Required fields in ai_integration
    REQUIRED_AI_INTEGRATION = [
        "temperature",
        "seed",
        "system_prompt",
        "user_prompt_template",
        "tool_choice",
        "max_iterations",
        "budget_usd",
        "provenance_tracking",
        "ai_summary"
    ]

    # Valid enum values
    VALID_DOMAINS = [
        "Domain1_Industrial",
        "Domain2_HVAC",
        "Domain3_CrossCutting"
    ]

    VALID_COMPLEXITY = ["Low", "Medium", "High"]

    VALID_PRIORITY = ["P0_Critical", "P1_High", "P2_Medium"]

    VALID_STATUS = [
        "Spec_Needed",
        "Spec_Complete",
        "In_Development",
        "Testing",
        "Production"
    ]

    VALID_TOOL_CATEGORIES = [
        "calculation",
        "lookup",
        "aggregation",
        "analysis",
        "optimization"
    ]

    def __init__(self, schema_path: Optional[str] = None):
        """Initialize validator with optional JSON schema."""
        self.schema = None
        if schema_path and Path(schema_path).exists():
            with open(schema_path, 'r') as f:
                self.schema = json.load(f)

    def validate_file(self, file_path: str) -> ValidationReport:
        """Validate a single agent spec file."""
        report = ValidationReport(file_path=file_path, is_valid=True)

        # Check file exists
        if not Path(file_path).exists():
            report.add_error("file", "existence", f"File not found: {file_path}")
            return report

        # Load YAML
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                spec = yaml.safe_load(f)
        except yaml.YAMLError as e:
            report.add_error("yaml", "parsing", f"YAML parsing error: {str(e)}")
            return report
        except Exception as e:
            report.add_error("file", "reading", f"Error reading file: {str(e)}")
            return report

        if not spec:
            report.add_error("yaml", "empty", "File is empty or invalid YAML")
            return report

        # Validate sections
        self._validate_required_sections(spec, report)
        self._validate_agent_metadata(spec.get("agent_metadata", {}), report)
        self._validate_description(spec.get("description", {}), report)
        self._validate_tools(spec.get("tools", {}), report)
        self._validate_ai_integration(spec.get("ai_integration", {}), report)
        self._validate_inputs(spec.get("inputs", {}), report)
        self._validate_outputs(spec.get("outputs", {}), report)
        self._validate_testing(spec.get("testing", {}), report)
        self._validate_deployment(spec.get("deployment", {}), report)
        self._validate_documentation(spec.get("documentation", {}), report)
        self._validate_compliance(spec.get("compliance", {}), report)
        self._validate_metadata(spec.get("metadata", {}), report)

        return report

    def _validate_required_sections(self, spec: Dict, report: ValidationReport):
        """Validate all required top-level sections exist."""
        for section in self.REQUIRED_SECTIONS:
            if section not in spec:
                report.add_error(
                    "structure",
                    section,
                    f"Required section '{section}' is missing"
                )

    def _validate_agent_metadata(self, metadata: Dict, report: ValidationReport):
        """Validate agent_metadata section."""
        section = "agent_metadata"

        # Check required fields
        for field in self.REQUIRED_AGENT_METADATA:
            if field not in metadata:
                report.add_error(section, field, f"Required field '{field}' is missing")

        # Validate domain
        if "domain" in metadata:
            domain = metadata["domain"]
            if domain not in self.VALID_DOMAINS:
                report.add_error(
                    section,
                    "domain",
                    f"Invalid domain value",
                    expected=", ".join(self.VALID_DOMAINS),
                    actual=domain
                )

        # Validate complexity
        if "complexity" in metadata:
            complexity = metadata["complexity"]
            if complexity not in self.VALID_COMPLEXITY:
                report.add_error(
                    section,
                    "complexity",
                    f"Invalid complexity value",
                    expected=", ".join(self.VALID_COMPLEXITY),
                    actual=complexity
                )

        # Validate priority
        if "priority" in metadata:
            priority = metadata["priority"]
            if priority not in self.VALID_PRIORITY:
                report.add_error(
                    section,
                    "priority",
                    f"Invalid priority value",
                    expected=", ".join(self.VALID_PRIORITY),
                    actual=priority
                )

        # Validate status
        if "status" in metadata:
            status = metadata["status"]
            if status not in self.VALID_STATUS:
                report.add_error(
                    section,
                    "status",
                    f"Invalid status value",
                    expected=", ".join(self.VALID_STATUS),
                    actual=status
                )

        # Validate version format (semver)
        if "version" in metadata:
            version = metadata["version"]
            if not re.match(r'^\d+\.\d+\.\d+$', str(version)):
                report.add_error(
                    section,
                    "version",
                    f"Version must follow semantic versioning (e.g., 1.0.0)",
                    actual=version
                )

        # Validate agent_id format
        if "agent_id" in metadata:
            agent_id = metadata["agent_id"]
            if not re.match(r'^[a-z_]+/[a-z_]+$', str(agent_id)):
                report.add_warning(
                    section,
                    "agent_id",
                    f"Agent ID should follow format 'domain/agent_name' (lowercase with underscores)",
                    actual=agent_id
                )

    def _validate_description(self, description: Dict, report: ValidationReport):
        """Validate description section."""
        section = "description"

        # Check required fields
        for field in self.REQUIRED_DESCRIPTION:
            if field not in description:
                report.add_error(section, field, f"Required field '{field}' is missing")

        # Validate strategic_context
        if "strategic_context" in description:
            context = description["strategic_context"]
            required_context_fields = ["global_impact", "opportunity", "market_size", "technology_maturity"]
            for field in required_context_fields:
                if field not in context:
                    report.add_warning(
                        section,
                        f"strategic_context.{field}",
                        f"Recommended field '{field}' is missing from strategic_context"
                    )

        # Validate key_capabilities is a list
        if "key_capabilities" in description:
            capabilities = description["key_capabilities"]
            if not isinstance(capabilities, list):
                report.add_error(
                    section,
                    "key_capabilities",
                    "key_capabilities must be a list"
                )
            elif len(capabilities) < 3:
                report.add_warning(
                    section,
                    "key_capabilities",
                    f"Consider adding more capabilities (currently: {len(capabilities)}, recommended: 3+)"
                )

    def _validate_tools(self, tools: Dict, report: ValidationReport):
        """Validate tools section - CRITICAL for tool-first design."""
        section = "tools"

        # Check tool_count exists
        if "tool_count" not in tools:
            report.add_error(section, "tool_count", "Required field 'tool_count' is missing")

        # Check tools_list exists
        if "tools_list" not in tools:
            report.add_error(section, "tools_list", "Required field 'tools_list' is missing")
            return

        tools_list = tools["tools_list"]
        if not isinstance(tools_list, list):
            report.add_error(section, "tools_list", "tools_list must be a list")
            return

        # Validate tool_count matches actual count
        tool_count = tools.get("tool_count", 0)
        actual_count = len(tools_list)
        if tool_count != actual_count:
            report.add_error(
                section,
                "tool_count",
                f"tool_count mismatch",
                expected=actual_count,
                actual=tool_count
            )

        # Validate recommended tool count range
        if actual_count < 4:
            report.add_warning(
                section,
                "tool_count",
                f"Tool count is low (typical: 4-12 tools)",
                actual=actual_count
            )
        elif actual_count > 12:
            report.add_warning(
                section,
                "tool_count",
                f"Tool count is high (typical: 4-12 tools) - consider splitting agent",
                actual=actual_count
            )

        # Validate each tool
        for i, tool in enumerate(tools_list):
            self._validate_tool(tool, i, report)

    def _validate_tool(self, tool: Dict, index: int, report: ValidationReport):
        """Validate individual tool definition."""
        section = f"tools.tools_list[{index}]"

        # Required tool fields
        required_fields = ["tool_id", "name", "description", "category", "deterministic", "parameters", "returns", "implementation"]
        for field in required_fields:
            if field not in tool:
                report.add_error(section, field, f"Required field '{field}' is missing")

        # Validate deterministic flag (CRITICAL!)
        if "deterministic" in tool:
            if tool["deterministic"] != True:
                report.add_error(
                    section,
                    "deterministic",
                    "All tools MUST be deterministic (deterministic: true)",
                    expected=True,
                    actual=tool["deterministic"]
                )

        # Validate category
        if "category" in tool:
            category = tool["category"]
            if category not in self.VALID_TOOL_CATEGORIES:
                report.add_error(
                    section,
                    "category",
                    f"Invalid tool category",
                    expected=", ".join(self.VALID_TOOL_CATEGORIES),
                    actual=category
                )

        # Validate parameters schema
        if "parameters" in tool:
            params = tool["parameters"]
            if not isinstance(params, dict):
                report.add_error(section, "parameters", "parameters must be an object/dict")
            else:
                if "type" not in params:
                    report.add_error(section, "parameters.type", "parameters must have 'type' field")
                if "properties" not in params:
                    report.add_error(section, "parameters.properties", "parameters must have 'properties' field")
                if "required" not in params:
                    report.add_warning(section, "parameters.required", "Consider adding 'required' field to parameters")

        # Validate returns schema
        if "returns" in tool:
            returns = tool["returns"]
            if not isinstance(returns, dict):
                report.add_error(section, "returns", "returns must be an object/dict")
            else:
                if "type" not in returns:
                    report.add_error(section, "returns.type", "returns must have 'type' field")
                if "properties" not in returns:
                    report.add_warning(section, "returns.properties", "Consider adding 'properties' field to returns")

        # Validate implementation
        if "implementation" in tool:
            impl = tool["implementation"]
            recommended_impl_fields = ["calculation_method", "data_source", "accuracy", "validation"]
            for field in recommended_impl_fields:
                if field not in impl:
                    report.add_warning(
                        section,
                        f"implementation.{field}",
                        f"Recommended field '{field}' is missing from implementation"
                    )

        # Validate example exists
        if "example" not in tool:
            report.add_warning(section, "example", "Consider adding an example for this tool")

    def _validate_ai_integration(self, ai_integration: Dict, report: ValidationReport):
        """Validate ai_integration section - CRITICAL for deterministic AI."""
        section = "ai_integration"

        # Check required fields
        for field in self.REQUIRED_AI_INTEGRATION:
            if field not in ai_integration:
                report.add_error(section, field, f"Required field '{field}' is missing")

        # Validate temperature=0.0 (CRITICAL!)
        if "temperature" in ai_integration:
            temp = ai_integration["temperature"]
            if temp != 0.0:
                report.add_error(
                    section,
                    "temperature",
                    "temperature MUST be 0.0 for deterministic results",
                    expected=0.0,
                    actual=temp
                )

        # Validate seed=42 (CRITICAL!)
        if "seed" in ai_integration:
            seed = ai_integration["seed"]
            if seed != 42:
                report.add_error(
                    section,
                    "seed",
                    "seed MUST be 42 for reproducibility",
                    expected=42,
                    actual=seed
                )

        # Validate provenance_tracking=true
        if "provenance_tracking" in ai_integration:
            if ai_integration["provenance_tracking"] != True:
                report.add_error(
                    section,
                    "provenance_tracking",
                    "provenance_tracking MUST be true",
                    expected=True,
                    actual=ai_integration["provenance_tracking"]
                )

        # Validate ai_summary=true
        if "ai_summary" in ai_integration:
            if ai_integration["ai_summary"] != True:
                report.add_warning(
                    section,
                    "ai_summary",
                    "ai_summary should be true for human-readable explanations",
                    expected=True,
                    actual=ai_integration["ai_summary"]
                )

        # Validate budget is reasonable
        if "budget_usd" in ai_integration:
            budget = ai_integration["budget_usd"]
            if budget > 1.0:
                report.add_warning(
                    section,
                    "budget_usd",
                    f"Budget seems high (typical: $0.10-$0.50)",
                    actual=budget
                )

        # Validate system_prompt exists and has content
        if "system_prompt" in ai_integration:
            prompt = ai_integration["system_prompt"]
            if not prompt or len(str(prompt).strip()) < 50:
                report.add_warning(
                    section,
                    "system_prompt",
                    "system_prompt should be comprehensive (at least 50 characters)"
                )

            # Check for critical instructions
            prompt_str = str(prompt).lower()
            critical_phrases = ["use tools", "never estimate", "never guess", "deterministic"]
            missing_phrases = [p for p in critical_phrases if p not in prompt_str]
            if missing_phrases:
                report.add_warning(
                    section,
                    "system_prompt",
                    f"system_prompt should emphasize: {', '.join(missing_phrases)}"
                )

    def _validate_inputs(self, inputs: Dict, report: ValidationReport):
        """Validate inputs section."""
        section = "inputs"

        if "input_schema" not in inputs:
            report.add_error(section, "input_schema", "Required field 'input_schema' is missing")
        else:
            schema = inputs["input_schema"]
            if "type" not in schema:
                report.add_error(section, "input_schema.type", "input_schema must have 'type' field")
            if "properties" not in schema:
                report.add_error(section, "input_schema.properties", "input_schema must have 'properties' field")

        if "example_input" not in inputs:
            report.add_warning(section, "example_input", "Consider adding example_input for documentation")

    def _validate_outputs(self, outputs: Dict, report: ValidationReport):
        """Validate outputs section."""
        section = "outputs"

        if "output_schema" not in outputs:
            report.add_error(section, "output_schema", "Required field 'output_schema' is missing")
        else:
            schema = outputs["output_schema"]
            if "type" not in schema:
                report.add_error(section, "output_schema.type", "output_schema must have 'type' field")
            if "properties" not in schema:
                report.add_error(section, "output_schema.properties", "output_schema must have 'properties' field")

            # Check for provenance in output
            if "properties" in schema:
                props = schema["properties"]
                if "provenance" not in props:
                    report.add_warning(
                        section,
                        "output_schema.properties.provenance",
                        "Consider adding 'provenance' field to output_schema for audit trail"
                    )

        if "example_output" not in outputs:
            report.add_warning(section, "example_output", "Consider adding example_output for documentation")

    def _validate_testing(self, testing: Dict, report: ValidationReport):
        """Validate testing section."""
        section = "testing"

        # Check test_coverage_target
        if "test_coverage_target" not in testing:
            report.add_error(section, "test_coverage_target", "Required field 'test_coverage_target' is missing")
        else:
            coverage = testing["test_coverage_target"]
            if coverage < 0.80:
                report.add_error(
                    section,
                    "test_coverage_target",
                    "test_coverage_target MUST be at least 0.80 (80%)",
                    expected=">=0.80",
                    actual=coverage
                )

        # Check test_categories
        if "test_categories" not in testing:
            report.add_error(section, "test_categories", "Required field 'test_categories' is missing")
        else:
            categories = testing["test_categories"]
            if not isinstance(categories, list):
                report.add_error(section, "test_categories", "test_categories must be a list")
            else:
                # Check for required test categories
                required_categories = ["unit_tests", "integration_tests", "determinism_tests", "boundary_tests"]
                category_names = [c.get("category", "") for c in categories]
                for req_cat in required_categories:
                    if req_cat not in category_names:
                        report.add_error(
                            section,
                            f"test_categories.{req_cat}",
                            f"Required test category '{req_cat}' is missing"
                        )

        # Check performance_requirements
        if "performance_requirements" not in testing:
            report.add_warning(section, "performance_requirements", "Consider adding performance_requirements")

    def _validate_deployment(self, deployment: Dict, report: ValidationReport):
        """Validate deployment section."""
        section = "deployment"

        required_fields = ["pack_id", "pack_version", "dependencies", "resource_requirements"]
        for field in required_fields:
            if field not in deployment:
                report.add_warning(section, field, f"Recommended field '{field}' is missing")

        # Validate dependencies
        if "dependencies" in deployment:
            deps = deployment["dependencies"]
            if "python_packages" not in deps:
                report.add_warning(section, "dependencies.python_packages", "Consider specifying python_packages")
            if "greenlang_modules" not in deps:
                report.add_warning(section, "dependencies.greenlang_modules", "Consider specifying greenlang_modules")

    def _validate_documentation(self, documentation: Dict, report: ValidationReport):
        """Validate documentation section."""
        section = "documentation"

        # Check boolean flags
        doc_flags = ["readme", "api_docs", "examples", "tutorials"]
        for flag in doc_flags:
            if flag not in documentation:
                report.add_warning(section, flag, f"Recommended field '{flag}' is missing")
            elif documentation.get(flag) != True:
                report.add_info(section, flag, f"{flag} is set to false - documentation may be incomplete")

    def _validate_compliance(self, compliance: Dict, report: ValidationReport):
        """Validate compliance section - CRITICAL for security."""
        section = "compliance"

        # Check zero_secrets (CRITICAL!)
        if "zero_secrets" not in compliance:
            report.add_error(section, "zero_secrets", "Required field 'zero_secrets' is missing")
        else:
            if compliance["zero_secrets"] != True:
                report.add_error(
                    section,
                    "zero_secrets",
                    "zero_secrets MUST be true (no hardcoded secrets)",
                    expected=True,
                    actual=compliance["zero_secrets"]
                )

        # Check SBOM requirement
        if "sbom_required" not in compliance:
            report.add_warning(section, "sbom_required", "Consider adding sbom_required field")

        # Check standards
        if "standards" not in compliance:
            report.add_warning(section, "standards", "Consider listing applicable standards")
        else:
            standards = compliance["standards"]
            if not isinstance(standards, list) or len(standards) == 0:
                report.add_info(section, "standards", "No compliance standards specified")

    def _validate_metadata(self, metadata: Dict, report: ValidationReport):
        """Validate metadata section."""
        section = "metadata"

        required_fields = ["created_date", "created_by", "last_modified", "review_status"]
        for field in required_fields:
            if field not in metadata:
                report.add_warning(section, field, f"Recommended field '{field}' is missing")

        # Validate date format
        date_fields = ["created_date", "last_modified"]
        for field in date_fields:
            if field in metadata:
                date_str = str(metadata[field])
                if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                    report.add_warning(
                        section,
                        field,
                        f"Date should be in YYYY-MM-DD format",
                        actual=date_str
                    )

    def validate_batch(self, directory: str, pattern: str = "**/*.yaml") -> List[ValidationReport]:
        """Validate all YAML files in directory matching pattern."""
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"Error: Directory not found: {directory}")
            return []

        yaml_files = list(dir_path.glob(pattern))
        reports = []

        print(f"\n🔍 Found {len(yaml_files)} YAML files to validate\n")

        for file_path in yaml_files:
            # Skip template file
            if "Template" in file_path.name or "template" in file_path.name:
                print(f"⏭️  Skipping template: {file_path.name}")
                continue

            print(f"📄 Validating: {file_path.name}...")
            report = self.validate_file(str(file_path))
            reports.append(report)

            # Print summary
            status = "✅ PASS" if report.is_valid else "❌ FAIL"
            print(f"   {status} - Errors: {len(report.errors)}, Warnings: {len(report.warnings)}")

        return reports


def print_batch_summary(reports: List[ValidationReport]):
    """Print summary of batch validation."""
    total = len(reports)
    passed = sum(1 for r in reports if r.is_valid)
    failed = total - passed
    total_errors = sum(len(r.errors) for r in reports)
    total_warnings = sum(len(r.warnings) for r in reports)

    print("\n" + "=" * 80)
    print("BATCH VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total files validated: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Total errors: {total_errors}")
    print(f"Total warnings: {total_warnings}")
    print("=" * 80)

    if failed > 0:
        print("\n❌ Failed files:")
        for report in reports:
            if not report.is_valid:
                print(f"   • {Path(report.file_path).name} ({len(report.errors)} errors)")
    else:
        print("\n✨ All files passed validation!")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate GreenLang AgentSpec V2.0 files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate single file
  python validate_agent_specs.py specs/domain1_industrial/industrial_process/agent_001_industrial_process_heat.yaml

  # Validate all specs in directory
  python validate_agent_specs.py --batch specs/

  # Validate with JSON schema
  python validate_agent_specs.py --schema schemas/agentspec_v2_schema.json specs/agent_001.yaml

  # Save report to file
  python validate_agent_specs.py agent_001.yaml --output report.txt
        """
    )

    parser.add_argument(
        "file_path",
        nargs="?",
        help="Path to agent spec YAML file to validate"
    )
    parser.add_argument(
        "--batch",
        metavar="DIR",
        help="Validate all YAML files in directory"
    )
    parser.add_argument(
        "--schema",
        metavar="PATH",
        help="Path to JSON schema file for validation"
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        help="Save validation report to file"
    )
    parser.add_argument(
        "--pattern",
        default="**/*.yaml",
        help="File pattern for batch mode (default: **/*.yaml)"
    )

    args = parser.parse_args()

    # Create validator
    validator = AgentSpecValidator(schema_path=args.schema)

    # Batch mode
    if args.batch:
        reports = validator.validate_batch(args.batch, args.pattern)
        print_batch_summary(reports)

        # Save detailed reports if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                for report in reports:
                    f.write(report.get_summary())
                    f.write("\n\n")
            print(f"\n📝 Detailed reports saved to: {args.output}")

        # Exit with error if any failed
        sys.exit(0 if all(r.is_valid for r in reports) else 1)

    # Single file mode
    elif args.file_path:
        report = validator.validate_file(args.file_path)
        summary = report.get_summary()
        print(summary)

        # Save report if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"\n📝 Report saved to: {args.output}")

        sys.exit(0 if report.is_valid else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
