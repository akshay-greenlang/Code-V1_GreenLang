# -*- coding: utf-8 -*-
"""
Violation Scanner Agent
========================

Scan codebase for policy violations and compliance issues.

Author: Monitoring & Observability Team
Created: 2025-11-09
"""

import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import re
import ast
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


@dataclass
class Violation:
    """Represents a policy violation"""
    id: str
    type: str
    severity: str  # info, warning, critical
    file_path: str
    line_number: int
    message: str
    application: str
    team: str
    detected_at: datetime
    auto_fixable: bool = False


class ViolationScanner:
    """
    Scans codebase for policy violations.
    """

    def __init__(self, codebase_path: str):
        self.codebase_path = Path(codebase_path)
        self.violations: List[Violation] = []
        self.violation_patterns = self._define_violation_patterns()

    def _define_violation_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Define patterns for detecting violations"""
        return {
            'custom_llm_wrapper': {
                'patterns': [
                    r'class\s+\w*LLM\w*\(.*\):',
                    r'def\s+call_openai\s*\(',
                    r'def\s+call_anthropic\s*\(',
                    r'openai\.ChatCompletion\.create',
                    r'from\s+openai\s+import\s+OpenAI',
                ],
                'severity': 'critical',
                'message': 'Custom LLM wrapper detected. Must use LLMClient infrastructure.',
                'auto_fixable': False
            },
            'custom_cache_implementation': {
                'patterns': [
                    r'class\s+\w*Cache\w*\(.*\):(?!.*SemanticCache)',
                    r'def\s+cache_result\s*\(',
                    r'@cache',
                ],
                'severity': 'warning',
                'message': 'Custom cache implementation. Consider using SemanticCache.',
                'auto_fixable': False
            },
            'missing_adr': {
                'patterns': [],  # Detected by logic, not regex
                'severity': 'warning',
                'message': 'Large custom code file without ADR documentation.',
                'auto_fixable': False
            },
            'hardcoded_credentials': {
                'patterns': [
                    r'api_key\s*=\s*["\'][^"\']+["\']',
                    r'password\s*=\s*["\'][^"\']+["\']',
                    r'secret\s*=\s*["\'][^"\']+["\']',
                    r'token\s*=\s*["\'][^"\']+["\']',
                ],
                'severity': 'critical',
                'message': 'Hardcoded credentials detected. Use environment variables.',
                'auto_fixable': False
            },
            'missing_error_handling': {
                'patterns': [
                    r'def\s+\w+\([^)]*\):\s*\n\s+(?!try:)',
                ],
                'severity': 'info',
                'message': 'Missing error handling. Consider adding try-except blocks.',
                'auto_fixable': True
            },
            'deprecated_imports': {
                'patterns': [
                    r'from\s+old_infrastructure\s+import',
                    r'import\s+deprecated_module',
                ],
                'severity': 'warning',
                'message': 'Using deprecated infrastructure imports.',
                'auto_fixable': True
            }
        }

    async def scan_codebase(self) -> List[Violation]:
        """Scan entire codebase for violations"""
        logger.info(f"Scanning codebase: {self.codebase_path}")

        python_files = list(self.codebase_path.rglob("*.py"))
        logger.info(f"Found {len(python_files)} Python files to scan")

        tasks = [self._scan_file(file_path) for file_path in python_files]
        await asyncio.gather(*tasks)

        logger.info(f"Scan completed. Found {len(self.violations)} violations")
        return self.violations

    async def _scan_file(self, file_path: Path) -> None:
        """Scan a single file for violations"""
        try:
            content = file_path.read_text(encoding='utf-8')

            # Determine application and team from path
            app_info = self._get_app_info(file_path)

            # Pattern-based scanning
            for violation_type, config in self.violation_patterns.items():
                for pattern in config['patterns']:
                    matches = re.finditer(pattern, content, re.MULTILINE)
                    for match in matches:
                        line_number = content[:match.start()].count('\n') + 1

                        violation = Violation(
                            id=f"V{DeterministicClock.now().strftime('%Y%m%d%H%M%S')}{len(self.violations)}",
                            type=violation_type,
                            severity=config['severity'],
                            file_path=str(file_path.relative_to(self.codebase_path)),
                            line_number=line_number,
                            message=config['message'],
                            application=app_info['application'],
                            team=app_info['team'],
                            detected_at=DeterministicClock.now(),
                            auto_fixable=config['auto_fixable']
                        )
                        self.violations.append(violation)

            # Logic-based checks
            await self._check_missing_adr(file_path, content, app_info)
            await self._check_test_coverage(file_path, content, app_info)

        except Exception as e:
            logger.error(f"Error scanning {file_path}: {e}")

    def _get_app_info(self, file_path: Path) -> Dict[str, str]:
        """Extract application and team info from file path"""
        path_str = str(file_path)

        if 'GL-CSRD-APP' in path_str:
            return {'application': 'csrd-reporting', 'team': 'csrd'}
        elif 'GL-VCCI-Carbon-APP' in path_str:
            return {'application': 'vcci-scope3', 'team': 'carbon'}
        elif 'greenlang/infrastructure' in path_str:
            return {'application': 'infrastructure', 'team': 'platform'}
        else:
            return {'application': 'unknown', 'team': 'unknown'}

    async def _check_missing_adr(
        self,
        file_path: Path,
        content: str,
        app_info: Dict[str, str]
    ) -> None:
        """Check if large custom code files have ADRs"""
        # Count custom code lines (exclude comments and infrastructure imports)
        lines = content.split('\n')
        custom_lines = 0

        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                # Check if line is custom code (not infrastructure)
                if not any(infra in stripped for infra in [
                    'from greenlang.infrastructure',
                    'from greenlang.core',
                    'import greenlang'
                ]):
                    custom_lines += 1

        # Check if ADR exists
        if custom_lines > 100:
            adr_path = file_path.parent / 'ADR' / f'{file_path.stem}.md'
            if not adr_path.exists():
                violation = Violation(
                    id=f"V{DeterministicClock.now().strftime('%Y%m%d%H%M%S')}{len(self.violations)}",
                    type='missing_adr',
                    severity='warning',
                    file_path=str(file_path.relative_to(self.codebase_path)),
                    line_number=1,
                    message=f'File has {custom_lines} lines of custom code without ADR.',
                    application=app_info['application'],
                    team=app_info['team'],
                    detected_at=DeterministicClock.now(),
                    auto_fixable=False
                )
                self.violations.append(violation)

    async def _check_test_coverage(
        self,
        file_path: Path,
        content: str,
        app_info: Dict[str, str]
    ) -> None:
        """Check if file has corresponding test file"""
        if 'test_' in file_path.name or '_test.py' in file_path.name:
            return  # Skip test files themselves

        # Look for corresponding test file
        test_path1 = file_path.parent / f'test_{file_path.name}'
        test_path2 = file_path.parent / 'tests' / f'test_{file_path.name}'

        if not test_path1.exists() and not test_path2.exists():
            # Check if file has testable functions
            try:
                tree = ast.parse(content)
                functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

                if len(functions) > 0:
                    violation = Violation(
                        id=f"V{DeterministicClock.now().strftime('%Y%m%d%H%M%S')}{len(self.violations)}",
                        type='missing_tests',
                        severity='info',
                        file_path=str(file_path.relative_to(self.codebase_path)),
                        line_number=1,
                        message=f'File has {len(functions)} functions but no test file.',
                        application=app_info['application'],
                        team=app_info['team'],
                        detected_at=DeterministicClock.now(),
                        auto_fixable=False
                    )
                    self.violations.append(violation)
            except Exception as e:
                logger.debug(f"Failed to parse AST for test coverage check in {file_path}: {e}")

    def generate_report(self) -> str:
        """Generate violation report"""
        violations_by_severity = {
            'critical': [v for v in self.violations if v.severity == 'critical'],
            'warning': [v for v in self.violations if v.severity == 'warning'],
            'info': [v for v in self.violations if v.severity == 'info']
        }

        violations_by_type = {}
        for v in self.violations:
            violations_by_type[v.type] = violations_by_type.get(v.type, 0) + 1

        report = f"""
Policy Violation Scan Report
=============================
Scan Time: {DeterministicClock.now().isoformat()}
Codebase: {self.codebase_path}

Summary:
--------
Total Violations: {len(self.violations)}
Critical: {len(violations_by_severity['critical'])}
Warning: {len(violations_by_severity['warning'])}
Info: {len(violations_by_severity['info'])}

Violations by Type:
-------------------
"""

        for vtype, count in sorted(violations_by_type.items(), key=lambda x: x[1], reverse=True):
            report += f"{vtype}: {count}\n"

        report += "\n\nCritical Violations (Require Immediate Action):\n"
        report += "=" * 50 + "\n"

        for v in violations_by_severity['critical'][:10]:  # Show top 10
            report += f"""
Type: {v.type}
File: {v.file_path}:{v.line_number}
Application: {v.application} (Team: {v.team})
Message: {v.message}
Auto-fixable: {v.auto_fixable}
---
"""

        return report

    async def export_to_database(self, db_connection: Any) -> None:
        """Export violations to database for tracking"""
        # In production: save to PostgreSQL, MongoDB, etc.
        logger.info(f"Exporting {len(self.violations)} violations to database")

        for violation in self.violations:
            # INSERT INTO violations ...
            pass

        logger.info("Export completed")


async def main():
    """Main entry point for violation scanner"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    codebase_path = "C:\\Users\\aksha\\Code-V1_GreenLang"
    scanner = ViolationScanner(codebase_path)

    # Scan codebase
    violations = await scanner.scan_codebase()

    # Generate report
    report = scanner.generate_report()
    print(report)

    # Save report to file
    report_path = Path(codebase_path) / "greenlang" / "monitoring" / "reports" / f"violations_{DeterministicClock.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)

    logger.info(f"Report saved to: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
