"""
GL-Agent-Factory: Registry Audit System

This module provides comprehensive auditing capabilities for the agent registry,
including duplicate detection, validation, and health reporting.

Usage:
    from agents.registry_audit import RegistryAuditor

    auditor = RegistryAuditor()
    report = auditor.full_audit()
    print(report.summary())
"""
import importlib
import logging
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict

from .registry import AGENT_DEFINITIONS, AgentInfo, AgentRegistry

logger = logging.getLogger(__name__)


@dataclass
class DuplicateIssue:
    """Represents a duplicate agent ID issue."""
    base_id: str
    conflicting_ids: List[str]
    agents: List[Dict[str, str]]
    severity: str  # "ERROR", "WARNING", "INFO"
    recommendation: str


@dataclass
class ValidationIssue:
    """Represents a validation issue with an agent definition."""
    agent_id: str
    field: str
    issue_type: str  # "MISSING", "INVALID", "INCONSISTENT"
    message: str
    severity: str


@dataclass
class ModuleIssue:
    """Represents an issue loading an agent module."""
    agent_id: str
    module_path: str
    error_type: str
    error_message: str


@dataclass
class AuditReport:
    """Complete audit report for the agent registry."""
    timestamp: str
    total_definitions: int
    unique_base_ids: int
    unique_full_ids: int

    # Issues found
    duplicate_issues: List[DuplicateIssue] = field(default_factory=list)
    validation_issues: List[ValidationIssue] = field(default_factory=list)
    module_issues: List[ModuleIssue] = field(default_factory=list)

    # Statistics
    by_category: Dict[str, int] = field(default_factory=dict)
    by_type: Dict[str, int] = field(default_factory=dict)
    by_priority: Dict[str, int] = field(default_factory=dict)
    by_status: Dict[str, int] = field(default_factory=dict)

    # Health
    loadable_count: int = 0
    failed_count: int = 0
    overall_health: str = "UNKNOWN"

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "GL-AGENT-FACTORY REGISTRY AUDIT REPORT",
            "=" * 70,
            f"Timestamp: {self.timestamp}",
            "",
            "OVERVIEW",
            "-" * 40,
            f"Total Definitions: {self.total_definitions}",
            f"Unique Base IDs (e.g., GL-001): {self.unique_base_ids}",
            f"Unique Full IDs (including B variants): {self.unique_full_ids}",
            f"Overall Health: {self.overall_health}",
            "",
        ]

        if self.duplicate_issues:
            lines.extend([
                "DUPLICATE ID ISSUES",
                "-" * 40,
            ])
            for issue in self.duplicate_issues:
                lines.append(f"  [{issue.severity}] Base ID: {issue.base_id}")
                for agent in issue.agents:
                    lines.append(f"    - {agent['id']}: {agent['name']} ({agent['module']})")
                lines.append(f"    Recommendation: {issue.recommendation}")
                lines.append("")

        if self.validation_issues:
            lines.extend([
                "VALIDATION ISSUES",
                "-" * 40,
            ])
            for issue in self.validation_issues:
                lines.append(f"  [{issue.severity}] {issue.agent_id}: {issue.message}")
            lines.append("")

        if self.module_issues:
            lines.extend([
                "MODULE LOADING ISSUES",
                "-" * 40,
            ])
            for issue in self.module_issues:
                lines.append(f"  {issue.agent_id}: {issue.error_type}")
                lines.append(f"    Module: {issue.module_path}")
                lines.append(f"    Error: {issue.error_message}")
            lines.append("")

        lines.extend([
            "STATISTICS BY CATEGORY",
            "-" * 40,
        ])
        for cat, count in sorted(self.by_category.items(), key=lambda x: -x[1]):
            lines.append(f"  {cat}: {count}")

        lines.extend([
            "",
            "STATISTICS BY PRIORITY",
            "-" * 40,
        ])
        for pri, count in sorted(self.by_priority.items()):
            lines.append(f"  {pri}: {count}")

        lines.extend([
            "",
            "MODULE HEALTH",
            "-" * 40,
            f"  Loadable: {self.loadable_count}",
            f"  Failed: {self.failed_count}",
            "",
            "=" * 70,
        ])

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "total_definitions": self.total_definitions,
            "unique_base_ids": self.unique_base_ids,
            "unique_full_ids": self.unique_full_ids,
            "duplicate_issues": [asdict(i) for i in self.duplicate_issues],
            "validation_issues": [asdict(i) for i in self.validation_issues],
            "module_issues": [asdict(i) for i in self.module_issues],
            "by_category": self.by_category,
            "by_type": self.by_type,
            "by_priority": self.by_priority,
            "by_status": self.by_status,
            "loadable_count": self.loadable_count,
            "failed_count": self.failed_count,
            "overall_health": self.overall_health,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class RegistryAuditor:
    """
    Comprehensive auditor for the GL-Agent-Factory registry.

    Features:
    - Duplicate ID detection (base and full IDs)
    - Definition validation (required fields, format)
    - Module loadability checks
    - Statistical analysis
    - Health scoring
    """

    # Required fields for agent definitions
    REQUIRED_FIELDS = ["agent_id", "agent_name", "module_path", "class_name", "category", "agent_type"]

    # Valid values for constrained fields
    VALID_PRIORITIES = ["P0", "P1", "P2", "P3"]
    VALID_COMPLEXITIES = ["Low", "Medium", "High"]
    VALID_STATUSES = ["Implemented", "In Progress", "Planned", "Deprecated"]

    def __init__(self):
        self.definitions = AGENT_DEFINITIONS

    def _extract_base_id(self, agent_id: str) -> str:
        """Extract base ID from variant IDs (e.g., GL-031B -> GL-031)."""
        # Handle variants like GL-031B, GL-031C, etc.
        if agent_id[-1].isalpha() and agent_id[-2] == "-":
            return agent_id[:-1]
        if agent_id[-1].isalpha() and agent_id[-2].isdigit():
            return agent_id[:-1]
        return agent_id

    def detect_duplicates(self) -> List[DuplicateIssue]:
        """Detect duplicate agent IDs in the registry."""
        issues = []

        # Group by base ID
        base_id_groups: Dict[str, List[AgentInfo]] = defaultdict(list)
        for agent in self.definitions:
            base_id = self._extract_base_id(agent.agent_id)
            base_id_groups[base_id].append(agent)

        # Find duplicates
        for base_id, agents in base_id_groups.items():
            if len(agents) > 1:
                # Check if these are intentional variants (with B, C suffixes)
                ids = [a.agent_id for a in agents]
                has_variants = any(id != base_id for id in ids)

                if has_variants:
                    # These appear to be intentional variants
                    severity = "WARNING"
                    recommendation = (
                        f"Consider restructuring: {base_id}-A for primary, "
                        f"{base_id}-B for secondary, or use unique base IDs"
                    )
                else:
                    # True duplicates - same ID for different agents
                    severity = "ERROR"
                    recommendation = (
                        "Assign unique IDs to these agents or merge them"
                    )

                issues.append(DuplicateIssue(
                    base_id=base_id,
                    conflicting_ids=ids,
                    agents=[
                        {
                            "id": a.agent_id,
                            "name": a.agent_name,
                            "module": a.module_path,
                            "category": a.category
                        }
                        for a in agents
                    ],
                    severity=severity,
                    recommendation=recommendation
                ))

        return sorted(issues, key=lambda x: (x.severity != "ERROR", x.base_id))

    def validate_definitions(self) -> List[ValidationIssue]:
        """Validate all agent definitions for completeness and correctness."""
        issues = []

        for agent in self.definitions:
            # Check required fields
            for field_name in self.REQUIRED_FIELDS:
                value = getattr(agent, field_name, None)
                if not value:
                    issues.append(ValidationIssue(
                        agent_id=agent.agent_id,
                        field=field_name,
                        issue_type="MISSING",
                        message=f"Required field '{field_name}' is empty or missing",
                        severity="ERROR"
                    ))

            # Validate ID format (GL-NNN or GL-NNNX)
            if not self._validate_id_format(agent.agent_id):
                issues.append(ValidationIssue(
                    agent_id=agent.agent_id,
                    field="agent_id",
                    issue_type="INVALID",
                    message=f"ID format should be GL-NNN or GL-NNNX (e.g., GL-001, GL-031B)",
                    severity="WARNING"
                ))

            # Validate priority
            if agent.priority and agent.priority not in self.VALID_PRIORITIES:
                issues.append(ValidationIssue(
                    agent_id=agent.agent_id,
                    field="priority",
                    issue_type="INVALID",
                    message=f"Invalid priority '{agent.priority}'. Valid: {self.VALID_PRIORITIES}",
                    severity="WARNING"
                ))

            # Validate complexity
            if agent.complexity and agent.complexity not in self.VALID_COMPLEXITIES:
                issues.append(ValidationIssue(
                    agent_id=agent.agent_id,
                    field="complexity",
                    issue_type="INVALID",
                    message=f"Invalid complexity '{agent.complexity}'. Valid: {self.VALID_COMPLEXITIES}",
                    severity="WARNING"
                ))

            # Validate status
            if agent.status and agent.status not in self.VALID_STATUSES:
                issues.append(ValidationIssue(
                    agent_id=agent.agent_id,
                    field="status",
                    issue_type="INVALID",
                    message=f"Invalid status '{agent.status}'. Valid: {self.VALID_STATUSES}",
                    severity="WARNING"
                ))

            # Check module path format
            if agent.module_path and not agent.module_path.startswith("gl_"):
                issues.append(ValidationIssue(
                    agent_id=agent.agent_id,
                    field="module_path",
                    issue_type="INCONSISTENT",
                    message=f"Module path should start with 'gl_' for consistency",
                    severity="INFO"
                ))

            # Check class name format (PascalCase ending in Agent)
            if agent.class_name and not agent.class_name.endswith("Agent"):
                issues.append(ValidationIssue(
                    agent_id=agent.agent_id,
                    field="class_name",
                    issue_type="INCONSISTENT",
                    message=f"Class name should end with 'Agent' for consistency",
                    severity="INFO"
                ))

        return issues

    def _validate_id_format(self, agent_id: str) -> bool:
        """Validate agent ID format."""
        import re
        # Matches GL-001, GL-100, GL-031B, etc.
        pattern = r"^GL-\d{3}[A-Z]?$"
        return bool(re.match(pattern, agent_id))

    def check_modules(self) -> Tuple[int, int, List[ModuleIssue]]:
        """Check if all agent modules can be loaded."""
        loadable = 0
        failed = 0
        issues = []

        seen_modules = set()

        for agent in self.definitions:
            # Skip already checked modules
            if agent.module_path in seen_modules:
                continue
            seen_modules.add(agent.module_path)

            try:
                importlib.import_module(f"backend.agents.{agent.module_path}")
                loadable += 1
            except ModuleNotFoundError as e:
                failed += 1
                issues.append(ModuleIssue(
                    agent_id=agent.agent_id,
                    module_path=agent.module_path,
                    error_type="MODULE_NOT_FOUND",
                    error_message=str(e)
                ))
            except ImportError as e:
                failed += 1
                issues.append(ModuleIssue(
                    agent_id=agent.agent_id,
                    module_path=agent.module_path,
                    error_type="IMPORT_ERROR",
                    error_message=str(e)
                ))
            except Exception as e:
                failed += 1
                issues.append(ModuleIssue(
                    agent_id=agent.agent_id,
                    module_path=agent.module_path,
                    error_type=type(e).__name__,
                    error_message=str(e)
                ))

        return loadable, failed, issues

    def compute_statistics(self) -> Dict[str, Dict[str, int]]:
        """Compute statistics about the registry."""
        stats = {
            "by_category": defaultdict(int),
            "by_type": defaultdict(int),
            "by_priority": defaultdict(int),
            "by_status": defaultdict(int),
        }

        seen_ids = set()
        for agent in self.definitions:
            if agent.agent_id in seen_ids:
                continue
            seen_ids.add(agent.agent_id)

            stats["by_category"][agent.category] += 1
            stats["by_type"][agent.agent_type] += 1
            stats["by_priority"][agent.priority] += 1
            stats["by_status"][agent.status] += 1

        return {k: dict(v) for k, v in stats.items()}

    def full_audit(self, check_modules: bool = False) -> AuditReport:
        """
        Perform a full audit of the agent registry.

        Args:
            check_modules: Whether to attempt importing modules (slower)

        Returns:
            Complete AuditReport with all findings
        """
        report = AuditReport(
            timestamp=datetime.utcnow().isoformat() + "Z",
            total_definitions=len(self.definitions),
            unique_base_ids=len(set(self._extract_base_id(a.agent_id) for a in self.definitions)),
            unique_full_ids=len(set(a.agent_id for a in self.definitions)),
        )

        # Detect duplicates
        report.duplicate_issues = self.detect_duplicates()

        # Validate definitions
        report.validation_issues = self.validate_definitions()

        # Compute statistics
        stats = self.compute_statistics()
        report.by_category = stats["by_category"]
        report.by_type = stats["by_type"]
        report.by_priority = stats["by_priority"]
        report.by_status = stats["by_status"]

        # Check modules if requested
        if check_modules:
            loadable, failed, module_issues = self.check_modules()
            report.loadable_count = loadable
            report.failed_count = failed
            report.module_issues = module_issues

        # Determine overall health
        error_count = sum(1 for i in report.duplicate_issues if i.severity == "ERROR")
        error_count += sum(1 for i in report.validation_issues if i.severity == "ERROR")
        error_count += len(report.module_issues)

        warning_count = sum(1 for i in report.duplicate_issues if i.severity == "WARNING")
        warning_count += sum(1 for i in report.validation_issues if i.severity == "WARNING")

        if error_count > 0:
            report.overall_health = "UNHEALTHY"
        elif warning_count > 10:
            report.overall_health = "DEGRADED"
        elif warning_count > 0:
            report.overall_health = "FAIR"
        else:
            report.overall_health = "HEALTHY"

        return report

    def generate_fix_recommendations(self) -> List[Dict[str, Any]]:
        """Generate specific fix recommendations for registry issues."""
        recommendations = []

        # Get duplicate issues
        duplicates = self.detect_duplicates()

        for issue in duplicates:
            if issue.severity == "ERROR":
                # True duplicates need new IDs
                recommendations.append({
                    "type": "DUPLICATE_ID",
                    "action": "REASSIGN_ID",
                    "base_id": issue.base_id,
                    "affected_agents": issue.agents,
                    "suggested_fix": f"Assign unique IDs: {issue.base_id}, {issue.base_id}A, {issue.base_id}B"
                })
            else:
                # Variants should be renamed for clarity
                recommendations.append({
                    "type": "VARIANT_IDS",
                    "action": "STANDARDIZE_NAMING",
                    "base_id": issue.base_id,
                    "affected_agents": issue.agents,
                    "suggested_fix": (
                        f"Standardize variant naming: Use {issue.base_id}-A, {issue.base_id}-B "
                        f"or allocate new base IDs from available range"
                    )
                })

        return recommendations


def run_audit(check_modules: bool = False, output_format: str = "text") -> str:
    """
    Run a full registry audit and return results.

    Args:
        check_modules: Whether to check module loadability
        output_format: "text" for human-readable, "json" for JSON

    Returns:
        Audit report as string
    """
    auditor = RegistryAuditor()
    report = auditor.full_audit(check_modules=check_modules)

    if output_format == "json":
        return report.to_json()
    return report.summary()


if __name__ == "__main__":
    import sys

    check_modules = "--check-modules" in sys.argv
    json_output = "--json" in sys.argv

    output_format = "json" if json_output else "text"
    print(run_audit(check_modules=check_modules, output_format=output_format))
