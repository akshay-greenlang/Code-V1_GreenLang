#!/usr/bin/env python3
"""
Automated test fixer for AGENT-EUDR-035 Improvement Plan Creator tests.

Fixes common patterns:
1. Method names: aggregate() → aggregate_findings(), analyze() → analyze_gaps(), etc.
2. Return types: AggregatedFindings has .findings list, not len()
3. Field names: article_reference → eudr_article_ref, gap_score → severity_score
"""
import re
import sys
from pathlib import Path

def fix_test_file(filepath: Path) -> int:
    """Fix common test patterns in a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    fixes = 0

    # Fix 1: aggregate_findings return type - change len(result) to len(result.findings)
    pattern1 = r'assert len\(result\) (==|>=|<=|>|<|!=) '
    if re.search(pattern1, content) and 'test_finding_aggregator' in filepath.name:
        content = re.sub(r'assert len\(result\) (==|>=|<=|>|<|!=) ', r'assert len(result.findings) \1 ', content)
        fixes += 1
        print(f"  - Fixed len(result) -> len(result.findings)")

    # Fix 2: result.findings iteration
    if 'test_finding_aggregator' in filepath.name:
        # result == [] -> result.findings == []
        content = re.sub(r'assert result == \[\]', 'assert result.findings == []', content)
        # for finding in result: -> for finding in result.findings:
        content = re.sub(r'for finding in result:', 'for finding in result.findings:', content)
        # isinstance(result, list) -> isinstance(result, AggregatedFindings)
        content = content.replace('isinstance(result, list)', 'isinstance(result, AggregatedFindings)')
        # r1[0].provenance_hash -> r1.provenance_hash
        content = re.sub(r'r1\[0\]\.provenance_hash', 'r1.provenance_hash', content)
        content = re.sub(r'r2\[0\]\.provenance_hash', 'r2.provenance_hash', content)
        fixes += 1

    # Fix 3: Field name: article_reference -> eudr_article_ref
    if 'article_reference' in content:
        content = content.replace('.article_reference', '.eudr_article_ref')
        fixes += 1
        print(f"  - Fixed article_reference -> eudr_article_ref")

    # Fix 4: Field name: gap_score -> severity_score
    if 'gap_score' in content:
        content = content.replace('.gap_score', '.severity_score')
        fixes += 1
        print(f"  - Fixed gap_score -> severity_score")

    # Fix 5: Method names in test_gap_analyzer.py
    if 'test_gap_analyzer' in filepath.name:
        content = re.sub(r'analyzer\.analyze\(', 'analyzer.analyze_gaps(', content)
        fixes += 1

    # Fix 6: Method names in test_action_generator.py
    if 'test_action_generator' in filepath.name:
        content = re.sub(r'generator\.generate\(', 'generator.generate_actions(', content)
        fixes += 1

    # Fix 7: Method names in test_root_cause_mapper.py
    if 'test_root_cause_mapper' in filepath.name:
        content = re.sub(r'mapper\.analyze\(', 'mapper.analyze_root_causes(', content)
        content = re.sub(r'mapper\.build_fishbone_diagram\(', 'mapper.build_fishbone(', content)
        fixes += 1

    # Fix 8: Method names in test_prioritization_engine.py
    if 'test_prioritization_engine' in filepath.name:
        content = re.sub(r'engine\.prioritize\(', 'engine.prioritize_actions(', content)
        fixes += 1

    # Fix 9: Method names in test_progress_tracker.py
    if 'test_progress_tracker' in filepath.name:
        content = re.sub(r'tracker\.track\(', 'tracker.capture_snapshot(', content)
        content = re.sub(r'tracker\.record_progress\(', 'tracker.capture_snapshot(', content)
        fixes += 1

    # Fix 10: Method names in test_stakeholder_coordinator.py
    if 'test_stakeholder_coordinator' in filepath.name:
        content = re.sub(r'coordinator\.assign\(', 'coordinator.assign_stakeholders(', content)
        content = re.sub(r'coordinator\.notify\(', 'coordinator.send_notification(', content)
        fixes += 1

    # Write back if changed
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return fixes
    return 0

def main():
    test_dir = Path(__file__).parent
    test_files = list(test_dir.glob('test_*.py'))

    print(f"Found {len(test_files)} test files")
    total_fixes = 0

    for test_file in test_files:
        print(f"\nProcessing {test_file.name}...")
        fixes = fix_test_file(test_file)
        total_fixes += fixes
        if fixes > 0:
            print(f"  Applied {fixes} fix patterns")

    print(f"\nTotal: {total_fixes} fix patterns applied")
    return 0

if __name__ == '__main__':
    sys.exit(main())
