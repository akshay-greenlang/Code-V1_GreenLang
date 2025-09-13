#!/usr/bin/env python3
"""
Evaluate agent results and determine CI pass/fail status.
This script processes agent output and enforces governance decisions.
"""

import json
import sys
import argparse
from typing import Dict, Any, List
from datetime import datetime


class ResultEvaluator:
    """Evaluates agent results and enforces governance policies."""
    
    # Agent criticality levels
    BLOCKING_AGENTS = [
        'SpecGuardian',
        'SecScan',
        'SupplyChainSentinel',
        'ExitBarAuditor'
    ]
    
    WARNING_AGENTS = [
        'CodeSentinel',
        'PolicyLinter',
        'PackQC'
    ]
    
    INFO_AGENTS = [
        'DeterminismAuditor'
    ]
    
    def __init__(self, result: Dict[str, Any]):
        self.result = result
        self.agent = result.get('agent', 'Unknown')
        self.status = result.get('status', 'ERROR')
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the agent result and determine actions."""
        evaluation = {
            'agent': self.agent,
            'status': self.status,
            'timestamp': datetime.now().isoformat(),
            'should_block': False,
            'severity': self._get_severity(),
            'actions': [],
            'summary': ''
        }
        
        # Determine if this should block the CI
        if self.agent in self.BLOCKING_AGENTS:
            if self.status in ['FAIL', 'NO_GO', 'ERROR']:
                evaluation['should_block'] = True
                evaluation['summary'] = f"🚫 {self.agent} check failed - blocking merge"
            else:
                evaluation['summary'] = f"✅ {self.agent} check passed"
        
        elif self.agent in self.WARNING_AGENTS:
            if self.status == 'FAIL':
                evaluation['summary'] = f"⚠️ {self.agent} found issues - review recommended"
            else:
                evaluation['summary'] = f"✅ {self.agent} check passed"
        
        else:  # INFO_AGENTS
            evaluation['summary'] = f"ℹ️ {self.agent} completed - {self.status}"
        
        # Extract specific issues and actions
        evaluation['actions'] = self._extract_actions()
        evaluation['issues'] = self._extract_issues()
        
        return evaluation
    
    def _get_severity(self) -> str:
        """Determine the severity level of the result."""
        if self.status == 'ERROR':
            return 'CRITICAL'
        
        if self.agent in self.BLOCKING_AGENTS:
            return 'HIGH' if self.status in ['FAIL', 'NO_GO'] else 'LOW'
        elif self.agent in self.WARNING_AGENTS:
            return 'MEDIUM' if self.status == 'FAIL' else 'LOW'
        else:
            return 'INFO'
    
    def _extract_actions(self) -> List[str]:
        """Extract required actions from the result."""
        actions = []
        
        # Agent-specific action extraction
        if self.agent == 'SpecGuardian':
            if 'autofix_suggestions' in self.result:
                for fix in self.result['autofix_suggestions']:
                    actions.append(f"Fix {fix['file']}: {fix['reason']}")
        
        elif self.agent == 'SecScan':
            if 'critical_issues' in self.result:
                for issue in self.result['critical_issues']:
                    actions.append(f"Security: {issue}")
        
        elif self.agent == 'ExitBarAuditor':
            if 'blocking_issues' in self.result:
                for issue in self.result['blocking_issues']:
                    actions.append(f"Exit Bar: {issue.get('remediation', issue)}")
        
        elif self.agent == 'CodeSentinel':
            if 'issues' in self.result:
                for issue in self.result.get('issues', []):
                    if issue.get('severity') == 'ERROR':
                        actions.append(f"Fix {issue['file']}:{issue['line']} - {issue['message']}")
        
        return actions
    
    def _extract_issues(self) -> List[Dict[str, Any]]:
        """Extract detailed issues from the result."""
        issues = []
        
        # Generic issue extraction
        for key in ['errors', 'critical_issues', 'blocking_issues', 'violations']:
            if key in self.result:
                for item in self.result[key]:
                    if isinstance(item, dict):
                        issues.append(item)
                    else:
                        issues.append({'description': str(item)})
        
        # Add warnings as non-blocking issues
        if 'warnings' in self.result:
            for warning in self.result['warnings']:
                if isinstance(warning, dict):
                    warning['severity'] = 'WARNING'
                    issues.append(warning)
                else:
                    issues.append({'description': str(warning), 'severity': 'WARNING'})
        
        return issues
    
    def format_github_comment(self) -> str:
        """Format result as a GitHub PR comment."""
        evaluation = self.evaluate()
        
        # Determine emoji based on status
        status_emoji = {
            'PASS': '✅',
            'GO': '✅',
            'FAIL': '❌',
            'NO_GO': '🚫',
            'ERROR': '⚠️',
            'WARN': '⚠️'
        }.get(self.status, '❓')
        
        comment = f"""
## {status_emoji} {self.agent} Result

**Status:** `{self.status}`
**Severity:** `{evaluation['severity']}`
**Should Block:** `{evaluation['should_block']}`

### Summary
{evaluation['summary']}
"""
        
        if evaluation['actions']:
            comment += "\n### Required Actions\n"
            for action in evaluation['actions']:
                comment += f"- [ ] {action}\n"
        
        if evaluation['issues']:
            comment += "\n### Issues Found\n"
            for issue in evaluation['issues'][:10]:  # Limit to 10 issues
                severity = issue.get('severity', 'INFO')
                desc = issue.get('description', issue)
                comment += f"- **{severity}**: {desc}\n"
        
        # Add agent-specific details
        if self.agent == 'PackQC' and 'quality_score' in self.result:
            comment += f"\n**Quality Score:** {self.result['quality_score']}/100\n"
        
        elif self.agent == 'ExitBarAuditor' and 'readiness_score' in self.result:
            comment += f"\n**Readiness Score:** {self.result['readiness_score']}/100\n"
        
        return comment


def main():
    parser = argparse.ArgumentParser(description='Evaluate agent results')
    parser.add_argument('result_file', help='Path to agent result JSON file')
    parser.add_argument('--format', choices=['json', 'github', 'console'],
                        default='console', help='Output format')
    parser.add_argument('--strict', action='store_true',
                        help='Treat warnings as failures')
    
    args = parser.parse_args()
    
    # Load result
    with open(args.result_file, 'r') as f:
        result = json.load(f)
    
    # Evaluate
    evaluator = ResultEvaluator(result)
    evaluation = evaluator.evaluate()
    
    # Apply strict mode if requested
    if args.strict and evaluation['severity'] in ['MEDIUM', 'HIGH', 'CRITICAL']:
        evaluation['should_block'] = True
    
    # Output based on format
    if args.format == 'json':
        print(json.dumps(evaluation, indent=2))
    
    elif args.format == 'github':
        print(evaluator.format_github_comment())
    
    else:  # console
        print(f"\n{evaluation['summary']}")
        print(f"Agent: {evaluation['agent']}")
        print(f"Status: {evaluation['status']}")
        print(f"Severity: {evaluation['severity']}")
        print(f"Blocking: {'YES' if evaluation['should_block'] else 'NO'}")
        
        if evaluation['actions']:
            print("\nRequired Actions:")
            for i, action in enumerate(evaluation['actions'], 1):
                print(f"  {i}. {action}")
        
        if evaluation['issues']:
            print(f"\nIssues Found: {len(evaluation['issues'])}")
            for issue in evaluation['issues'][:5]:
                print(f"  - {issue}")
    
    # Exit with appropriate code
    if evaluation['should_block']:
        print("\n❌ CI Check Failed - Blocking merge")
        sys.exit(1)
    else:
        print("\n✅ CI Check Passed")
        sys.exit(0)


if __name__ == '__main__':
    main()