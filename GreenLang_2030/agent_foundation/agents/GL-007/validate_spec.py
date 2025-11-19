"""
Validation script for GL-007 FurnacePerformanceMonitor Agent Specification.

Validates against AgentSpec V2.0 requirements:
- 11 mandatory sections
- All tools deterministic
- Complete JSON schemas
- Zero secrets compliance
- Documentation completeness
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime


class AgentSpecValidator:
    """Validates agent specifications against AgentSpec V2.0 requirements."""

    MANDATORY_SECTIONS = [
        "agent_metadata",
        "description",
        "tools",
        "ai_integration",
        "sub_agents",
        "inputs",
        "outputs",
        "testing",
        "deployment",
        "documentation",
        "compliance",
        "metadata"
    ]

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []
        self.spec_data = None

    def validate(self, spec_path: Path) -> Dict[str, Any]:
        """Validate agent specification."""
        print(f"Validating: {spec_path}")
        print("=" * 80)

        # Load YAML
        try:
            with open(spec_path, 'r', encoding='utf-8') as f:
                self.spec_data = yaml.safe_load(f)
            self.info.append("Successfully loaded YAML specification")
        except Exception as e:
            self.errors.append(f"Failed to load YAML: {str(e)}")
            return self._generate_report()

        # Run validation checks
        self._validate_mandatory_sections()
        self._validate_agent_metadata()
        self._validate_tools()
        self._validate_ai_integration()
        self._validate_sub_agents()
        self._validate_inputs_outputs()
        self._validate_testing()
        self._validate_deployment()
        self._validate_documentation()
        self._validate_compliance()
        self._validate_metadata()
        self._validate_line_count(spec_path)

        return self._generate_report()

    def _validate_mandatory_sections(self):
        """Validate all 11 mandatory sections are present."""
        print("\n[1/11] Validating Mandatory Sections...")
        missing_sections = []
        for section in self.MANDATORY_SECTIONS:
            if section not in self.spec_data:
                missing_sections.append(section)

        if missing_sections:
            self.errors.append(f"Missing mandatory sections: {', '.join(missing_sections)}")
        else:
            self.info.append(f"✓ All 11 mandatory sections present")

    def _validate_agent_metadata(self):
        """Validate agent_metadata section."""
        print("[2/11] Validating Agent Metadata...")
        if "agent_metadata" not in self.spec_data:
            return

        metadata = self.spec_data["agent_metadata"]
        required_fields = [
            "agent_id", "name", "version", "category", "domain",
            "type", "complexity", "priority"
        ]

        missing = [f for f in required_fields if f not in metadata]
        if missing:
            self.errors.append(f"agent_metadata missing fields: {', '.join(missing)}")
        else:
            self.info.append(f"✓ agent_metadata complete")

        # Check specific values
        if metadata.get("agent_id") != "GL-007":
            self.errors.append(f"agent_id should be 'GL-007', got '{metadata.get('agent_id')}'")

        if metadata.get("priority") != "P0":
            self.warnings.append(f"priority is '{metadata.get('priority')}', expected 'P0'")

        if metadata.get("complexity") == "High":
            self.info.append(f"✓ complexity: High (as required)")

    def _validate_tools(self):
        """Validate tools section."""
        print("[3/11] Validating Tools...")
        if "tools" not in self.spec_data:
            return

        tools_section = self.spec_data["tools"]

        # Check tool architecture
        if "tool_architecture" in tools_section:
            arch = tools_section["tool_architecture"]
            if arch.get("deterministic") != True:
                self.errors.append("tool_architecture.deterministic must be true")
            else:
                self.info.append("✓ tool_architecture.deterministic: true")

        # Check tools list
        if "tools_list" not in tools_section:
            self.errors.append("tools.tools_list is missing")
            return

        tools = tools_section["tools_list"]
        tool_count = len(tools)

        if tool_count < 8:
            self.warnings.append(f"Only {tool_count} tools defined (recommend 8-12)")
        elif tool_count >= 12:
            self.info.append(f"✓ {tool_count} tools defined (exceeds 8-10 standard)")
        else:
            self.info.append(f"✓ {tool_count} tools defined")

        # Validate each tool
        for idx, tool in enumerate(tools, 1):
            tool_id = tool.get("tool_id", f"tool_{idx}")

            # Check deterministic flag
            if tool.get("deterministic") != True:
                self.errors.append(f"Tool '{tool_id}' must be deterministic: true")

            # Check required fields
            required = ["tool_id", "name", "category", "deterministic", "description", "parameters", "returns", "implementation"]
            missing = [f for f in required if f not in tool]
            if missing:
                self.errors.append(f"Tool '{tool_id}' missing: {', '.join(missing)}")

            # Check parameters schema
            if "parameters" in tool:
                params = tool["parameters"]
                if not isinstance(params, dict) or "type" not in params:
                    self.errors.append(f"Tool '{tool_id}' parameters must be valid JSON schema")

            # Check returns schema
            if "returns" in tool:
                returns = tool["returns"]
                if not isinstance(returns, dict) or "type" not in returns:
                    self.errors.append(f"Tool '{tool_id}' returns must be valid JSON schema")

            # Check implementation
            if "implementation" in tool:
                impl = tool["implementation"]
                if "standards" not in impl:
                    self.warnings.append(f"Tool '{tool_id}' implementation missing standards")
                if "accuracy" not in impl:
                    self.warnings.append(f"Tool '{tool_id}' implementation missing accuracy")

        self.info.append(f"✓ Validated {tool_count} tools")

    def _validate_ai_integration(self):
        """Validate ai_integration section."""
        print("[4/11] Validating AI Integration...")
        if "ai_integration" not in self.spec_data:
            return

        ai = self.spec_data["ai_integration"]
        config = ai.get("configuration", {})

        # Check critical settings
        if config.get("temperature") != 0.0:
            self.errors.append(f"temperature must be 0.0 (deterministic), got {config.get('temperature')}")
        else:
            self.info.append("✓ temperature: 0.0 (deterministic)")

        if config.get("seed") != 42:
            self.errors.append(f"seed must be 42 (reproducible), got {config.get('seed')}")
        else:
            self.info.append("✓ seed: 42 (reproducible)")

        if config.get("provenance_tracking") != True:
            self.errors.append("provenance_tracking must be true")
        else:
            self.info.append("✓ provenance_tracking: true")

        # Check budget
        budget = config.get("budget_usd", 0)
        if budget <= 0.50:
            self.info.append(f"✓ budget_usd: {budget} (within limits)")
        else:
            self.warnings.append(f"budget_usd: {budget} exceeds typical $0.50 limit")

    def _validate_sub_agents(self):
        """Validate sub_agents section."""
        print("[5/11] Validating Sub-Agents...")
        if "sub_agents" not in self.spec_data:
            self.warnings.append("sub_agents section missing (acceptable if no coordination)")
            return

        sub_agents = self.spec_data["sub_agents"]

        if "coordination_architecture" in sub_agents:
            self.info.append("✓ coordination_architecture defined")

        # Count coordinating agents
        coord_count = 0
        for key in ["upstream_coordination", "peer_coordination", "downstream_coordination"]:
            if key in sub_agents:
                coord_count += len(sub_agents[key])

        if coord_count > 0:
            self.info.append(f"✓ Coordinates with {coord_count} agents")

    def _validate_inputs_outputs(self):
        """Validate inputs and outputs sections."""
        print("[6/11] Validating Inputs/Outputs...")

        # Validate inputs
        if "inputs" not in self.spec_data:
            self.errors.append("inputs section missing")
        else:
            inputs = self.spec_data["inputs"]
            if "schema" not in inputs:
                self.errors.append("inputs.schema missing")
            else:
                if inputs["schema"].get("type") != "object":
                    self.errors.append("inputs.schema must be type: object")
                else:
                    self.info.append("✓ inputs.schema complete")

        # Validate outputs
        if "outputs" not in self.spec_data:
            self.errors.append("outputs section missing")
        else:
            outputs = self.spec_data["outputs"]
            if "schema" not in outputs:
                self.errors.append("outputs.schema missing")
            else:
                if outputs["schema"].get("type") != "object":
                    self.errors.append("outputs.schema must be type: object")
                else:
                    self.info.append("✓ outputs.schema complete")

            # Check quality guarantees
            if "quality_guarantees" in outputs:
                guarantees = outputs["quality_guarantees"]
                if len(guarantees) >= 4:
                    self.info.append(f"✓ {len(guarantees)} quality guarantees defined")

    def _validate_testing(self):
        """Validate testing section."""
        print("[7/11] Validating Testing...")
        if "testing" not in self.spec_data:
            self.errors.append("testing section missing")
            return

        testing = self.spec_data["testing"]

        # Check coverage target
        coverage = testing.get("test_coverage_target", 0)
        if coverage < 0.85:
            self.errors.append(f"test_coverage_target {coverage} below 0.85 minimum")
        elif coverage >= 0.90:
            self.info.append(f"✓ test_coverage_target: {coverage} (exceeds 0.85 standard)")
        else:
            self.info.append(f"✓ test_coverage_target: {coverage}")

        # Check test categories
        if "test_categories" in testing:
            categories = testing["test_categories"]
            total_tests = sum(cat.get("count", 0) for cat in categories)
            if total_tests >= 60:
                self.info.append(f"✓ {total_tests} tests planned (exceeds minimum)")
            else:
                self.warnings.append(f"Only {total_tests} tests planned")

        # Check performance requirements
        if "performance_requirements" in testing:
            perf = testing["performance_requirements"]
            if "max_latency_ms" in perf:
                latency = perf.get("max_latency_ms", {})
                if "optimization" in latency:
                    opt_latency = latency["optimization"]
                    if opt_latency <= 3000:
                        self.info.append(f"✓ optimization max_latency_ms: {opt_latency} (exceeds 5000 standard)")

            if "max_cost_usd" in perf:
                cost = perf.get("max_cost_usd", {})
                if "per_optimization" in cost:
                    opt_cost = cost["per_optimization"]
                    if opt_cost <= 0.08:
                        self.info.append(f"✓ per_optimization max_cost_usd: {opt_cost} (exceeds 0.50 standard)")

            if "accuracy_targets" in perf:
                accuracy = perf["accuracy_targets"]
                min_accuracy = min(accuracy.values()) if accuracy else 0
                if min_accuracy >= 0.98:
                    self.info.append(f"✓ minimum accuracy target: {min_accuracy}")

    def _validate_deployment(self):
        """Validate deployment section."""
        print("[8/11] Validating Deployment...")
        if "deployment" not in self.spec_data:
            self.errors.append("deployment section missing")
            return

        deployment = self.spec_data["deployment"]

        # Check resource requirements
        if "resource_requirements" in deployment:
            resources = deployment["resource_requirements"]
            required = ["memory_mb", "cpu_cores"]
            missing = [f for f in required if f not in resources]
            if missing:
                self.warnings.append(f"deployment.resource_requirements missing: {', '.join(missing)}")
            else:
                mem = resources.get("memory_mb", 0)
                cpu = resources.get("cpu_cores", 0)
                self.info.append(f"✓ resource_requirements: {mem}Mi RAM, {cpu} CPU cores")

        # Check dependencies
        if "dependencies" in deployment:
            deps = deployment["dependencies"]
            if "python_packages" in deps:
                pkg_count = len(deps["python_packages"])
                self.info.append(f"✓ {pkg_count} Python dependencies defined")

    def _validate_documentation(self):
        """Validate documentation section."""
        print("[9/11] Validating Documentation...")
        if "documentation" not in self.spec_data:
            self.errors.append("documentation section missing")
            return

        docs = self.spec_data["documentation"]

        # Check readme sections
        if "readme_sections" in docs:
            sections = docs["readme_sections"]
            if len(sections) >= 11:
                self.info.append(f"✓ {len(sections)} README sections (exceeds minimum)")
            else:
                self.warnings.append(f"Only {len(sections)} README sections")

        # Check example use cases
        if "example_use_cases" in docs:
            examples = docs["example_use_cases"]
            if len(examples) >= 5:
                self.info.append(f"✓ {len(examples)} example use cases")
            else:
                self.warnings.append(f"Only {len(examples)} example use cases (recommend 5+)")

    def _validate_compliance(self):
        """Validate compliance section."""
        print("[10/11] Validating Compliance...")
        if "compliance" not in self.spec_data:
            self.errors.append("compliance section missing")
            return

        compliance = self.spec_data["compliance"]

        # Check zero_secrets
        if compliance.get("zero_secrets") != True:
            self.errors.append("compliance.zero_secrets must be true")
        else:
            self.info.append("✓ zero_secrets: true")

        # Check SBOM
        if "sbom" in compliance:
            self.info.append("✓ SBOM defined")

        # Check standards
        if "standards_compliance" in compliance:
            standards = compliance["standards_compliance"]
            standard_count = len(standards)
            self.info.append(f"✓ {standard_count} standards compliance documented")

        # Check security grade
        if "security_compliance" in compliance:
            sec = compliance["security_compliance"]
            if sec.get("security_grade") == "A+":
                self.info.append("✓ security_grade: A+")

    def _validate_metadata(self):
        """Validate metadata section."""
        print("[11/11] Validating Metadata...")
        if "metadata" not in self.spec_data:
            self.errors.append("metadata section missing")
            return

        metadata = self.spec_data["metadata"]

        required = ["specification_version", "created_date", "review_status"]
        missing = [f for f in required if f not in metadata]
        if missing:
            self.warnings.append(f"metadata missing: {', '.join(missing)}")

        # Check review status
        if metadata.get("review_status") == "Approved":
            self.info.append("✓ review_status: Approved")
        else:
            self.warnings.append(f"review_status: {metadata.get('review_status')} (not Approved)")

        # Check version control
        if "change_log" in metadata:
            changelog = metadata["change_log"]
            self.info.append(f"✓ change_log with {len(changelog)} entries")

    def _validate_line_count(self, spec_path: Path):
        """Validate specification line count."""
        print("\nValidating Line Count...")
        try:
            with open(spec_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                line_count = len(lines)

            if 1500 <= line_count <= 2000:
                self.info.append(f"✓ Line count: {line_count} (within 1,500-2,000 target)")
            elif line_count > 2000:
                self.info.append(f"✓ Line count: {line_count} (exceeds target, comprehensive)")
            else:
                self.warnings.append(f"Line count: {line_count} (below 1,500 target)")
        except Exception as e:
            self.warnings.append(f"Could not count lines: {str(e)}")

    def _generate_report(self) -> Dict[str, Any]:
        """Generate validation report."""
        status = "PASS" if len(self.errors) == 0 else "FAIL"

        report = {
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "spec_version_detected": self.spec_data.get("metadata", {}).get("specification_version", "unknown") if self.spec_data else "unknown",
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
            "summary": {
                "total_errors": len(self.errors),
                "total_warnings": len(self.warnings),
                "sections_validated": len(self.MANDATORY_SECTIONS)
            }
        }

        return report


def main():
    """Run validation."""
    spec_path = Path(__file__).parent / "agent_007_furnace_performance_monitor.yaml"

    validator = AgentSpecValidator()
    report = validator.validate(spec_path)

    # Print results
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print(f"Status: {report['status']}")
    print(f"Errors: {report['summary']['total_errors']}")
    print(f"Warnings: {report['summary']['total_warnings']}")
    print(f"Sections Validated: {report['summary']['sections_validated']}/11")

    if report['errors']:
        print("\nERRORS:")
        for error in report['errors']:
            print(f"  ✗ {error}")

    if report['warnings']:
        print("\nWARNINGS:")
        for warning in report['warnings']:
            print(f"  ⚠ {warning}")

    print("\nINFO:")
    for info_msg in report['info']:
        print(f"  {info_msg}")

    # Save report
    report_path = Path(__file__).parent / "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nDetailed report saved to: {report_path}")

    return 0 if report['status'] == "PASS" else 1


if __name__ == "__main__":
    exit(main())
