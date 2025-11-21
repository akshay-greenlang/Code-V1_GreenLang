#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code Health Check for GreenLang v0.2.0 Release Gate
"""

import os
import re
import json
import sys
from pathlib import Path

def check_version_consistency():
    """Check version consistency across files"""
    issues = []
    version_files = {
        'VERSION': None,
        'pyproject.toml': None,
        'greenlang/_version.py': None,
    }

    # Read VERSION file
    version_file = Path('VERSION')
    if version_file.exists():
        version_files['VERSION'] = version_file.read_text().strip()
    else:
        issues.append({
            'file': 'VERSION',
            'line': 0,
            'severity': 'ERROR',
            'category': 'version',
            'message': 'VERSION file not found',
            'fix': 'Create VERSION file with current version'
        })

    # Check pyproject.toml
    pyproject = Path('pyproject.toml')
    if pyproject.exists():
        content = pyproject.read_text()
        match = re.search(r'version\s*=\s*"([^"]+)"', content)
        if match:
            version_files['pyproject.toml'] = match.group(1)

    # Check _version.py
    version_py = Path('greenlang/_version.py')
    if version_py.exists():
        content = version_py.read_text()
        match = re.search(r'__version__\s*=\s*"([^"]+)"', content)
        if match:
            version_files['greenlang/_version.py'] = match.group(1)

    # Compare versions
    versions = [v for v in version_files.values() if v]
    if len(set(versions)) > 1:
        for file, version in version_files.items():
            if version and version != versions[0]:
                issues.append({
                    'file': file,
                    'line': 0,
                    'severity': 'ERROR',
                    'category': 'version',
                    'message': f'Version mismatch: {version} != {versions[0]}',
                    'fix': f'Update version to {versions[0]}'
                })

    return issues

def check_missing_setup_py():
    """Check if setup.py is missing (Week 0 requirement)"""
    issues = []
    if not Path('setup.py').exists():
        issues.append({
            'file': 'setup.py',
            'line': 0,
            'severity': 'ERROR',
            'category': 'setup',
            'message': 'setup.py file missing (required by Week 0 checklist)',
            'fix': 'Create setup.py for legacy compatibility'
        })
    return issues

def check_requirements():
    """Check Python version requirements"""
    issues = []

    # Check requirements.txt
    req_file = Path('requirements.txt')
    if req_file.exists():
        content = req_file.read_text().strip()
        if not content:
            issues.append({
                'file': 'requirements.txt',
                'line': 0,
                'severity': 'WARNING',
                'category': 'requirements',
                'message': 'requirements.txt is empty',
                'fix': 'Add package dependencies or remove the file'
            })

    # Check pytest.ini for python version
    pytest_ini = Path('pytest.ini')
    if pytest_ini.exists():
        content = pytest_ini.read_text()
        if 'python_version' not in content and 'required_plugins' not in content:
            issues.append({
                'file': 'pytest.ini',
                'line': 0,
                'severity': 'WARNING',
                'category': 'requirements',
                'message': 'pytest.ini missing Python version requirement',
                'fix': 'Add python_version or required_plugins specification'
            })

    return issues

def check_code_quality():
    """Check for code quality issues"""
    issues = []

    for root, dirs, files in os.walk('greenlang'):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.splitlines()

                    for i, line in enumerate(lines, 1):
                        # Check for hardcoded paths
                        if re.search(r'C:\\\\|C:/|/home/\w+/|/Users/\w+/', line) and not line.strip().startswith('#'):
                            issues.append({
                                'file': str(filepath),
                                'line': i,
                                'severity': 'ERROR',
                                'category': 'portability',
                                'message': 'Hardcoded non-portable path detected',
                                'fix': 'Use os.path.join() or pathlib.Path()'
                            })

                        # Check for print statements
                        if re.match(r'^\s*print\(', line) and not line.strip().startswith('#'):
                            issues.append({
                                'file': str(filepath),
                                'line': i,
                                'severity': 'WARNING',
                                'category': 'style',
                                'message': 'print() statement found',
                                'fix': 'Use logging instead of print()'
                            })

                        # Check for bare except
                        if re.match(r'^\s*except\s*:', line):
                            issues.append({
                                'file': str(filepath),
                                'line': i,
                                'severity': 'ERROR',
                                'category': 'dangerous',
                                'message': 'Bare except clause',
                                'fix': 'Specify exception type or use except Exception:'
                            })

                        # Check for TODO/FIXME
                        if re.search(r'#\s*(TODO|FIXME|XXX|HACK)', line):
                            issues.append({
                                'file': str(filepath),
                                'line': i,
                                'severity': 'WARNING',
                                'category': 'style',
                                'message': f'Unresolved {re.search(r"(TODO|FIXME|XXX|HACK)", line).group(1)} comment',
                                'fix': 'Resolve or create tracking issue'
                            })
                except Exception as e:
                    issues.append({
                        'file': str(filepath),
                        'line': 0,
                        'severity': 'WARNING',
                        'category': 'lint',
                        'message': f'Failed to parse file: {e}',
                        'fix': 'Check file encoding and syntax'
                    })

    return issues

def check_imports():
    """Check for import issues"""
    issues = []

    # Check for deprecated core.greenlang imports
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.splitlines()

                    for i, line in enumerate(lines, 1):
                        if re.search(r'from\s+core\.greenlang', line) and 'core/greenlang/__init__.py' not in str(filepath):
                            issues.append({
                                'file': str(filepath),
                                'line': i,
                                'severity': 'WARNING',
                                'category': 'import',
                                'message': 'Using deprecated core.greenlang import',
                                'fix': 'Change to: from greenlang import ...'
                            })
                except:
                    pass

    return issues

def main():
    """Run all health checks"""
    all_issues = []

    print("Running GreenLang Code Health Check...")
    print("=" * 60)

    # Run checks
    all_issues.extend(check_version_consistency())
    all_issues.extend(check_missing_setup_py())
    all_issues.extend(check_requirements())
    all_issues.extend(check_code_quality())
    all_issues.extend(check_imports())

    # Sort issues by severity
    errors = [i for i in all_issues if i['severity'] == 'ERROR']
    warnings = [i for i in all_issues if i['severity'] == 'WARNING']

    # Generate report
    status = 'PASS' if len(errors) == 0 else 'FAIL'

    report = {
        'status': status,
        'issues': all_issues[:50],  # Limit to first 50 issues
        'summary': f'Found {len(errors)} errors and {len(warnings)} warnings'
    }

    # Print report
    print(json.dumps(report, indent=2))

    # Exit with appropriate code
    sys.exit(0 if status == 'PASS' else 1)

if __name__ == '__main__':
    main()