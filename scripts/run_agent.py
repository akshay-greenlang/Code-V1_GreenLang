#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run GreenLang governance agents with collected context.
This script invokes the appropriate agent and returns structured results.
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any
import anthropic
from datetime import datetime
from greenlang.determinism import DeterministicClock


# Agent prompt templates
AGENT_PROMPTS = {
    'SpecGuardian': """
    You are GL-SpecGuardian. Analyze the following manifest changes for GreenLang spec v1.0 compliance.
    
    Context:
    {context}
    
    Validate all manifest files against the specification and return your assessment in the required JSON format.
    """,
    
    'CodeSentinel': """
    You are GL-CodeSentinel. Review the following code changes for quality issues.
    
    Context:
    {context}
    
    Analyze the diff and linting results, then return your assessment in the required JSON format.
    """,
    
    'SecScan': """
    You are GL-SecScan. Perform security analysis on the following changes.
    
    Context:
    {context}
    
    Scan for secrets, vulnerabilities, and policy violations. Return your findings in the required format.
    """,
    
    'PolicyLinter': """
    You are GL-PolicyLinter. Audit the following Rego policy changes.
    
    Context:
    {context}
    
    Check for security compliance, egress controls, and deny-by-default readiness.
    """,
    
    'SupplyChainSentinel': """
    You are GL-SupplyChainSentinel. Validate the following supply chain artifacts.
    
    Context:
    {context}
    
    Verify SBOM, signatures, and provenance. Return PASS/FAIL with detailed findings.
    """,
    
    'DeterminismAuditor': """
    You are GL-DeterminismAuditor. Analyze the following runs for deterministic behavior.
    
    Context:
    {context}
    
    Compare hashes and identify any sources of non-determinism.
    """,
    
    'PackQC': """
    You are GL-PackQC. Validate the following pack for quality standards.
    
    Context:
    {context}
    
    Check dependencies, resources, metadata, and compatibility. Return quality score and recommendations.
    """,
    
    'ExitBarAuditor': """
    You are GL-ExitBarAuditor. Evaluate production readiness against exit criteria.
    
    Context:
    {context}
    
    Assess quality, security, performance, and operational readiness. Return GO/NO_GO decision.
    """
}


class AgentRunner:
    """Runner for GreenLang governance agents."""
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.client = self._init_client()
    
    def _init_client(self) -> anthropic.Anthropic:
        """Initialize Anthropic client."""
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            # For testing, return mock client
            return None
        return anthropic.Anthropic(api_key=api_key)
    
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent with given context."""
        if not self.client:
            # Return mock response for testing
            return self._mock_response()
        
        prompt = AGENT_PROMPTS[self.agent_type].format(
            context=json.dumps(context, indent=2)
        )
        
        try:
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse agent response
            result = self._parse_response(response.content[0].text)
            return result
            
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e),
                'agent': self.agent_type,
                'timestamp': DeterministicClock.now().isoformat()
            }
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse agent response into structured format."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback to text analysis
                return {
                    'status': 'PASS' if 'PASS' in response_text else 'FAIL',
                    'raw_response': response_text,
                    'agent': self.agent_type
                }
        except json.JSONDecodeError:
            return {
                'status': 'ERROR',
                'error': 'Failed to parse agent response',
                'raw_response': response_text,
                'agent': self.agent_type
            }
    
    def _mock_response(self) -> Dict[str, Any]:
        """Generate mock response for testing."""
        mock_responses = {
            'SpecGuardian': {
                'status': 'PASS',
                'errors': [],
                'warnings': [],
                'spec_version_detected': '1.0.0'
            },
            'CodeSentinel': {
                'status': 'PASS',
                'issues': [],
                'summary': 'Code quality checks passed'
            },
            'SecScan': {
                'status': 'PASS',
                'findings': [],
                'summary': 'No security issues detected'
            },
            'PolicyLinter': {
                'status': 'PASS',
                'critical_violations': [],
                'warnings': []
            },
            'SupplyChainSentinel': {
                'status': 'PASS',
                'sbom': 'PASS',
                'signatures': 'PASS',
                'provenance': 'PASS'
            },
            'DeterminismAuditor': {
                'status': 'PASS',
                'hash_mismatches': [],
                'deterministic': True
            },
            'PackQC': {
                'status': 'PASS',
                'quality_score': 85,
                'publish_ready': True
            },
            'ExitBarAuditor': {
                'status': 'GO',
                'readiness_score': 92,
                'blocking_issues': []
            }
        }
        
        response = mock_responses.get(self.agent_type, {'status': 'PASS'})
        response['agent'] = self.agent_type
        response['timestamp'] = DeterministicClock.now().isoformat()
        response['mock'] = True
        
        return response


def main():
    parser = argparse.ArgumentParser(description='Run GreenLang governance agent')
    parser.add_argument('--agent', required=True, choices=AGENT_PROMPTS.keys(),
                        help='Agent type to run')
    parser.add_argument('--context', required=True, type=str,
                        help='Path to context JSON file')
    parser.add_argument('--mock', action='store_true',
                        help='Use mock responses for testing')
    
    args = parser.parse_args()
    
    # Load context
    with open(args.context, 'r') as f:
        context = json.load(f)
    
    # Run agent
    runner = AgentRunner(args.agent)
    
    if args.mock:
        # Force mock mode
        runner.client = None
    
    result = runner.run(context)
    
    # Output result
    print(json.dumps(result, indent=2))
    
    # Set exit code based on status
    if result.get('status') in ['FAIL', 'NO_GO', 'ERROR']:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == '__main__':
    main()