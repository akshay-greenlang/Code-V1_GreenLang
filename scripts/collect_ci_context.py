#!/usr/bin/env python3
"""
Collect CI context for GreenLang governance agents.
This script gathers relevant information based on the agent type.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List
import argparse


def run_command(cmd: List[str]) -> str:
    """Run a shell command and return output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"


def get_changed_files() -> List[str]:
    """Get list of changed files in the PR."""
    # In GitHub Actions, this would use the GitHub API
    # For now, use git diff
    try:
        base_branch = os.environ.get('GITHUB_BASE_REF', 'main')
        files = run_command(['git', 'diff', '--name-only', f'{base_branch}...HEAD'])
        return files.split('\n') if files else []
    except Exception:
        # Fallback to uncommitted changes
        files = run_command(['git', 'diff', '--name-only'])
        staged = run_command(['git', 'diff', '--cached', '--name-only'])
        all_files = set(files.split('\n') if files else [])
        all_files.update(staged.split('\n') if staged else [])
        return list(all_files)


def get_diff() -> str:
    """Get the full diff of changes."""
    try:
        base_branch = os.environ.get('GITHUB_BASE_REF', 'main')
        return run_command(['git', 'diff', f'{base_branch}...HEAD'])
    except Exception:
        # Fallback to uncommitted changes
        return run_command(['git', 'diff', 'HEAD'])


def collect_spec_guardian_context() -> Dict[str, Any]:
    """Collect context for SpecGuardian agent."""
    changed_files = get_changed_files()
    manifest_files = [f for f in changed_files if f.endswith(('.yaml', '.yml', '.json'))]
    
    context = {
        'changed_files': changed_files,
        'manifest_files': manifest_files,
        'manifests': {}
    }
    
    for file in manifest_files:
        if Path(file).exists():
            with open(file, 'r') as f:
                context['manifests'][file] = f.read()
    
    return context


def collect_code_sentinel_context() -> Dict[str, Any]:
    """Collect context for CodeSentinel agent."""
    diff = get_diff()
    
    # Run linting tools
    lint_results = {}
    
    # Python linting
    if any(f.endswith('.py') for f in get_changed_files()):
        lint_results['flake8'] = run_command(['flake8', '--format=json', '.'])
        lint_results['mypy'] = run_command(['mypy', '.'])
    
    # JavaScript/TypeScript linting
    if any(f.endswith(('.js', '.ts', '.tsx')) for f in get_changed_files()):
        lint_results['eslint'] = run_command(['eslint', '--format=json', '.'])
    
    return {
        'diff': diff,
        'changed_files': get_changed_files(),
        'lint_results': lint_results
    }


def collect_secscan_context() -> Dict[str, Any]:
    """Collect context for SecScan agent."""
    diff = get_diff()
    
    # Run security scans
    security_results = {}
    
    # Secret scanning (using trufflehog or similar)
    try:
        security_results['secrets'] = run_command(['trufflehog', 'filesystem', '.', '--json'])
    except Exception:
        security_results['secrets'] = "No secret scanner available"
    
    # Dependency scanning
    if Path('package.json').exists():
        security_results['npm_audit'] = run_command(['npm', 'audit', '--json'])
    
    if Path('requirements.txt').exists():
        security_results['pip_audit'] = run_command(['pip-audit', '--format', 'json'])
    
    return {
        'diff': diff,
        'changed_files': get_changed_files(),
        'security_scan': security_results
    }


def collect_policy_linter_context() -> Dict[str, Any]:
    """Collect context for PolicyLinter agent."""
    changed_files = get_changed_files()
    rego_files = [f for f in changed_files if f.endswith('.rego')]
    
    context = {
        'changed_files': changed_files,
        'rego_files': rego_files,
        'policies': {}
    }
    
    for file in rego_files:
        if Path(file).exists():
            with open(file, 'r') as f:
                context['policies'][file] = f.read()
    
    return context


def collect_supply_chain_context() -> Dict[str, Any]:
    """Collect context for SupplyChainSentinel agent."""
    context = {
        'sbom_exists': Path('sbom.spdx.json').exists(),
        'signatures': {},
        'provenance': {}
    }
    
    if context['sbom_exists']:
        with open('sbom.spdx.json', 'r') as f:
            context['sbom'] = json.load(f)
    
    # Check for cosign signatures
    try:
        context['cosign_verify'] = run_command(['cosign', 'verify', '--help'])
    except Exception:
        context['cosign_verify'] = "Cosign not available"
    
    return context


def collect_determinism_context() -> Dict[str, Any]:
    """Collect context for DeterminismAuditor agent."""
    run_files = list(Path('.').glob('**/run*.json'))
    
    context = {
        'run_files': [str(f) for f in run_files],
        'runs': {}
    }
    
    for run_file in run_files[:2]:  # Compare first two runs
        with open(run_file, 'r') as f:
            context['runs'][str(run_file)] = json.load(f)
    
    return context


def collect_packqc_context() -> Dict[str, Any]:
    """Collect context for PackQC agent."""
    pack_file = Path('pack.yaml')
    
    context = {
        'pack_exists': pack_file.exists(),
        'pack_size_mb': 0,
        'dependencies': []
    }
    
    if pack_file.exists():
        with open(pack_file, 'r') as f:
            context['pack_content'] = f.read()
        
        # Calculate pack size
        total_size = sum(f.stat().st_size for f in Path('.').rglob('*') if f.is_file())
        context['pack_size_mb'] = total_size / (1024 * 1024)
    
    return context


def collect_exitbar_context() -> Dict[str, Any]:
    """Collect context for ExitBarAuditor agent."""
    context = {
        'test_results': run_command(['pytest', '--json-report', '--json-report-file=/tmp/report.json']),
        'coverage': run_command(['coverage', 'report', '--format=json']),
        'security_scan': collect_secscan_context(),
        'performance_metrics': {},
        'approvals': []
    }
    
    # Check for approvals (would integrate with GitHub API)
    try:
        context['pr_approved'] = os.environ.get('PR_APPROVED', 'false') == 'true'
    except Exception:
        context['pr_approved'] = False
    
    return context


AGENT_COLLECTORS = {
    'SpecGuardian': collect_spec_guardian_context,
    'CodeSentinel': collect_code_sentinel_context,
    'SecScan': collect_secscan_context,
    'PolicyLinter': collect_policy_linter_context,
    'SupplyChainSentinel': collect_supply_chain_context,
    'DeterminismAuditor': collect_determinism_context,
    'PackQC': collect_packqc_context,
    'ExitBarAuditor': collect_exitbar_context,
}


def main():
    parser = argparse.ArgumentParser(description='Collect CI context for GreenLang agents')
    parser.add_argument('--agent', required=True, choices=AGENT_COLLECTORS.keys(),
                        help='Agent type to collect context for')
    args = parser.parse_args()
    
    collector = AGENT_COLLECTORS[args.agent]
    context = collector()
    
    # Add common context
    context['timestamp'] = run_command(['date', '--iso-8601=seconds'])
    context['commit'] = run_command(['git', 'rev-parse', 'HEAD'])
    context['branch'] = run_command(['git', 'branch', '--show-current'])
    context['agent'] = args.agent
    
    # Output JSON context
    print(json.dumps(context, indent=2))


if __name__ == '__main__':
    main()