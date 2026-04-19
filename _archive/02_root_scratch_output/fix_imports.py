#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch fix incorrect imports in EUDR-035 test files.
"""
import re
from pathlib import Path

# Map of incorrect -> correct imports
IMPORT_FIXES = {
    "ActionItem": "ImprovementAction",
    "ComplianceDomain": None,  # Remove, doesn't exist
    "FindingCategory": None,  # Remove, doesn't exist
    "FindingRecord": "Finding",
    "FindingSeverity": "GapSeverity",
    "FindingStatus": None,  # Remove, doesn't exist
    "GapAnalysisResult": None,  # Remove, doesn't exist
    "GapRecord": "ComplianceGap",
    "GapStatus": None,  # Remove, doesn't exist
    "GapType": None,  # Remove, doesn't exist
    "ImprovementPlanStatus": "PlanStatus",
    "MilestoneRecord": "ProgressMilestone",
    "MilestoneStatus": None,  # Remove, doesn't exist
    "PriorityLevel": None,  # Remove, doesn't exist
    "PriorityScore": None,  # Remove, doesn't exist
    "ProgressRecord": "ProgressSnapshot",
    "ProgressStatus": None,  # Remove, doesn't exist
    "RootCauseCategory": "FishboneCategory",
    "RootCauseStatus": None,  # Remove, doesn't exist
    "StakeholderRecord": None,  # Remove, doesn't exist
    "StakeholderRole": "RACIRole",
    "StakeholderStatus": None,  # Remove, doesn't exist
}

# Additional imports that might be needed
ADDITIONAL_IMPORTS = [
    "ActionStatus",
    "ActionType",
    "AggregatedFindings",
    "ComplianceGap",
    "EisenhowerQuadrant",
    "Finding",
    "FindingSource",
    "FishboneAnalysis",
    "FishboneCategory",
    "GapSeverity",
    "ImprovementAction",
    "ImprovementPlan",
    "NotificationChannel",
    "NotificationRecord",
    "PlanStatus",
    "PlanSummary",
    "PlanReport",
    "ProgressMilestone",
    "ProgressSnapshot",
    "RACIRole",
    "RiskLevel",
    "RootCause",
    "StakeholderAssignment",
]

def fix_imports_in_file(file_path: Path):
    """Fix imports in a single test file."""
    print(f"Fixing {file_path.name}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Find the models import block
    import_pattern = r'from greenlang\.agents\.eudr\.improvement_plan_creator\.models import \((.*?)\)'

    match = re.search(import_pattern, content, re.DOTALL)
    if not match:
        print(f"  No models import found in {file_path.name}")
        return False

    imports_block = match.group(1)
    imports = [imp.strip().strip(',') for imp in imports_block.split('\n') if imp.strip() and not imp.strip().startswith('#')]

    # Fix each import
    fixed_imports = set()
    for imp in imports:
        imp = imp.strip()
        if not imp:
            continue
        if imp in IMPORT_FIXES:
            if IMPORT_FIXES[imp]:  # Has a replacement
                fixed_imports.add(IMPORT_FIXES[imp])
            # else: skip (removed import)
        else:
            fixed_imports.add(imp)

    # Build new import block
    sorted_imports = sorted(fixed_imports)
    new_imports_block = "from greenlang.agents.eudr.improvement_plan_creator.models import (\n"
    for imp in sorted_imports:
        new_imports_block += f"    {imp},\n"
    new_imports_block += ")"

    # Replace in content
    content = re.sub(import_pattern, new_imports_block, content, flags=re.DOTALL)

    # Now fix usage in the code
    for old, new in IMPORT_FIXES.items():
        if new:  # Only replace if there's a replacement
            # Match word boundaries to avoid partial matches
            content = re.sub(rf'\b{old}\b', new, content)

    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  OK Fixed {file_path.name}")
        return True
    else:
        print(f"  No changes needed for {file_path.name}")
        return False

def main():
    test_dir = Path(__file__).parent / "tests" / "agents" / "eudr" / "improvement_plan_creator"

    test_files = [
        "test_action_generator.py",
        "test_api.py",
        "test_finding_aggregator.py",
        "test_gap_analyzer.py",
        "test_prioritization_engine.py",
        "test_progress_tracker.py",
        "test_root_cause_mapper.py",
        "test_stakeholder_coordinator.py",
    ]

    fixed_count = 0
    for test_file in test_files:
        file_path = test_dir / test_file
        if file_path.exists():
            if fix_imports_in_file(file_path):
                fixed_count += 1
        else:
            print(f"WARNING File not found: {test_file}")

    print(f"\nOK Fixed {fixed_count} files")

if __name__ == "__main__":
    main()
