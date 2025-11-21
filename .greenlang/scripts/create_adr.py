#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GreenLang ADR (Architecture Decision Record) Generator

Interactive tool to create Architecture Decision Records with validation
that infrastructure alternatives were considered.
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from greenlang.determinism import DeterministicClock
from greenlang.determinism import sorted_listdir


class ADRGenerator:
    """Generates Architecture Decision Records."""

    def __init__(self, adr_dir: str = None):
        if adr_dir is None:
            adr_dir = os.path.join(os.getcwd(), '.greenlang', 'adr')

        self.adr_dir = adr_dir
        os.makedirs(self.adr_dir, exist_ok=True)

        # Load existing ADRs
        self.existing_adrs = self._load_existing_adrs()

    def _load_existing_adrs(self) -> List[Dict]:
        """Load existing ADRs to get next number."""
        adrs = []

        if os.path.exists(self.adr_dir):
            for file in sorted_listdir(self.adr_dir):
                if file.startswith('ADR-') and file.endswith('.md'):
                    # Extract number
                    try:
                        num = int(file.split('-')[1])
                        adrs.append({'number': num, 'file': file})
                    except (ValueError, IndexError):
                        pass

        return sorted(adrs, key=lambda x: x['number'])

    def get_next_number(self) -> int:
        """Get next ADR number."""
        if self.existing_adrs:
            return self.existing_adrs[-1]['number'] + 1
        return 1

    def interactive_create(self) -> Dict:
        """Interactive questionnaire for ADR creation."""
        print("\n" + "=" * 80)
        print("GreenLang Architecture Decision Record Generator")
        print("=" * 80)
        print("\nThis tool will help you create an ADR for custom code implementations.")
        print("Please answer the following questions:\n")

        adr_data = {}

        # Title
        adr_data['title'] = input("1. Title of the decision: ").strip()

        # Status
        print("\n2. Status:")
        print("   - proposed: Decision is being considered")
        print("   - accepted: Decision has been approved")
        print("   - rejected: Decision was not approved")
        print("   - deprecated: Decision is no longer valid")
        print("   - superseded: Decision has been replaced")

        status = input("   Status [proposed]: ").strip() or "proposed"
        adr_data['status'] = status

        # Context
        print("\n3. Context:")
        print("   Describe the problem or situation that led to this decision.")
        print("   (Press Enter twice when done)")

        context_lines = []
        while True:
            line = input("   ")
            if not line:
                break
            context_lines.append(line)

        adr_data['context'] = '\n'.join(context_lines)

        # Infrastructure alternatives considered
        print("\n4. Did you consider GreenLang infrastructure alternatives?")
        print("   Examples:")
        print("   - greenlang.intelligence.ChatSession for LLM calls")
        print("   - greenlang.sdk.base.Agent for agent classes")
        print("   - greenlang.cache.CacheManager for caching")
        print("   - greenlang.validation.ValidationFramework for validation")

        considered = input("   Alternatives considered [yes/no]: ").strip().lower()

        if considered not in ['yes', 'y']:
            print("\n   ⚠️  WARNING: You must consider GreenLang infrastructure alternatives!")
            print("   Please review available infrastructure components before proceeding.")

            proceed = input("   Continue anyway? [yes/no]: ").strip().lower()
            if proceed not in ['yes', 'y']:
                print("   ADR creation cancelled.")
                sys.exit(0)

        print("\n5. What GreenLang infrastructure alternatives did you consider?")
        print("   (List each alternative. Press Enter twice when done)")

        alternatives = []
        while True:
            line = input("   - ").strip()
            if not line:
                break
            alternatives.append(line)

        adr_data['alternatives'] = alternatives

        if not alternatives:
            print("\n   ⚠️  WARNING: No alternatives listed!")

        # Justification
        print("\n6. Justification:")
        print("   Explain WHY the GreenLang infrastructure cannot be used.")
        print("   Valid reasons:")
        print("   - Infrastructure doesn't support required feature")
        print("   - Performance requirements exceed infrastructure capabilities")
        print("   - Third-party integration not supported by infrastructure")
        print("   (Press Enter twice when done)")

        justification_lines = []
        while True:
            line = input("   ")
            if not line:
                break
            justification_lines.append(line)

        adr_data['justification'] = '\n'.join(justification_lines)

        # Decision
        print("\n7. Decision:")
        print("   What was decided? Be specific about the implementation.")
        print("   (Press Enter twice when done)")

        decision_lines = []
        while True:
            line = input("   ")
            if not line:
                break
            decision_lines.append(line)

        adr_data['decision'] = '\n'.join(decision_lines)

        # Consequences
        print("\n8. Consequences:")
        print("   What are the positive and negative consequences?")
        print("   (Press Enter twice when done)")

        consequences_lines = []
        while True:
            line = input("   ")
            if not line:
                break
            consequences_lines.append(line)

        adr_data['consequences'] = '\n'.join(consequences_lines)

        # Future migration plan
        print("\n9. Future migration plan:")
        print("   How/when will this be migrated to GreenLang infrastructure?")

        migration_plan = input("   Plan: ").strip()
        adr_data['migration_plan'] = migration_plan

        # Reviewers
        print("\n10. Reviewers (comma-separated):")
        reviewers = input("   Names: ").strip()
        adr_data['reviewers'] = [r.strip() for r in reviewers.split(',') if r.strip()]

        return adr_data

    def generate_adr_content(self, adr_data: Dict, number: int) -> str:
        """Generate ADR markdown content."""

        # Validate alternatives were provided
        alternatives_section = ""
        if adr_data['alternatives']:
            alternatives_section = "\n### GreenLang Infrastructure Alternatives Considered\n\n"
            for alt in adr_data['alternatives']:
                alternatives_section += f"- {alt}\n"
        else:
            alternatives_section = "\n### GreenLang Infrastructure Alternatives Considered\n\n"
            alternatives_section += "⚠️ **WARNING**: No infrastructure alternatives were documented!\n\n"

        # Justification validation
        justification_section = f"\n### Justification for Custom Implementation\n\n{adr_data['justification']}\n"

        if len(adr_data['justification']) < 50:
            justification_section += "\n⚠️ **WARNING**: Justification seems insufficient!\n"

        # Migration plan
        migration_section = ""
        if adr_data.get('migration_plan'):
            migration_section = f"\n### Future Migration Plan\n\n{adr_data['migration_plan']}\n"

        # Reviewers
        reviewers_section = ""
        if adr_data.get('reviewers'):
            reviewers_section = "\n### Reviewers\n\n"
            for reviewer in adr_data['reviewers']:
                reviewers_section += f"- [ ] {reviewer}\n"

        content = f"""# ADR-{number:03d}: {adr_data['title']}

**Status**: {adr_data['status']}

**Date**: {DeterministicClock.now().strftime('%Y-%m-%d')}

## Context

{adr_data['context']}
{alternatives_section}
{justification_section}

## Decision

{adr_data['decision']}

## Consequences

{adr_data['consequences']}
{migration_section}
{reviewers_section}

---

**Compliance Checklist**:
- [ ] GreenLang infrastructure alternatives were thoroughly evaluated
- [ ] Custom implementation is absolutely necessary
- [ ] Migration plan to infrastructure is documented
- [ ] ADR has been reviewed and approved
- [ ] Implementation follows GreenLang coding standards

**Infrastructure Usage Policy**: All custom implementations must have a documented ADR
explaining why GreenLang infrastructure cannot be used. Failure to provide adequate
justification may result in code review rejection.

"""

        return content

    def create_adr(self, adr_data: Dict) -> str:
        """Create ADR file."""
        number = self.get_next_number()

        # Generate filename
        title_slug = adr_data['title'].lower()
        title_slug = ''.join(c if c.isalnum() or c in [' ', '-'] else '' for c in title_slug)
        title_slug = '-'.join(title_slug.split())

        filename = f"ADR-{number:03d}-{title_slug}.md"
        file_path = os.path.join(self.adr_dir, filename)

        # Generate content
        content = self.generate_adr_content(adr_data, number)

        # Write file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return file_path

    def list_adrs(self) -> str:
        """List all ADRs."""
        lines = []
        lines.append("=" * 80)
        lines.append("Architecture Decision Records")
        lines.append("=" * 80)
        lines.append("")

        if not self.existing_adrs:
            lines.append("No ADRs found.")
        else:
            for adr in self.existing_adrs:
                lines.append(f"- {adr['file']}")

        return '\n'.join(lines)

    def validate_adr(self, file_path: str) -> List[str]:
        """Validate an ADR file."""
        issues = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for required sections
            required_sections = [
                'Context',
                'GreenLang Infrastructure Alternatives Considered',
                'Justification for Custom Implementation',
                'Decision',
                'Consequences'
            ]

            for section in required_sections:
                if f"## {section}" not in content and f"### {section}" not in content:
                    issues.append(f"Missing required section: {section}")

            # Check for warnings
            if "WARNING" in content:
                issues.append("ADR contains warnings that should be addressed")

            # Check if checklist is complete
            if "- [ ]" in content:
                issues.append("Compliance checklist is not complete")

            # Check justification length
            if "Justification for Custom Implementation" in content:
                # Extract justification section
                start = content.find("Justification for Custom Implementation")
                end = content.find("##", start + 1)
                justification = content[start:end] if end != -1 else content[start:]

                if len(justification) < 200:
                    issues.append("Justification seems too short (< 200 characters)")

        except Exception as e:
            issues.append(f"Error reading ADR: {e}")

        return issues


def main():
    parser = argparse.ArgumentParser(
        description="GreenLang ADR Generator - Create Architecture Decision Records"
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all existing ADRs'
    )

    parser.add_argument(
        '--validate',
        help='Validate an ADR file'
    )

    parser.add_argument(
        '--adr-dir',
        help='ADR directory (default: .greenlang/adr)'
    )

    parser.add_argument(
        '--non-interactive',
        action='store_true',
        help='Non-interactive mode (provide data via JSON file)'
    )

    parser.add_argument(
        '--data',
        help='JSON file with ADR data (for non-interactive mode)'
    )

    args = parser.parse_args()

    # Initialize generator
    generator = ADRGenerator(adr_dir=args.adr_dir)

    # List ADRs
    if args.list:
        print(generator.list_adrs())
        sys.exit(0)

    # Validate ADR
    if args.validate:
        print(f"Validating {args.validate}...")
        issues = generator.validate_adr(args.validate)

        if issues:
            print("\nValidation issues found:")
            for issue in issues:
                print(f"  ✗ {issue}")
            sys.exit(1)
        else:
            print("✓ ADR is valid")
            sys.exit(0)

    # Create ADR
    if args.non_interactive and args.data:
        import json
        with open(args.data, 'r') as f:
            adr_data = json.load(f)
    else:
        adr_data = generator.interactive_create()

    # Generate ADR
    file_path = generator.create_adr(adr_data)

    print("\n" + "=" * 80)
    print("✓ ADR Created Successfully!")
    print("=" * 80)
    print(f"\nFile: {file_path}")
    print(f"\nNext steps:")
    print(f"1. Review the ADR file")
    print(f"2. Get approval from reviewers")
    print(f"3. Update checklist in the ADR")
    print(f"4. Commit the ADR to version control")
    print(f"\nValidate with: python create_adr.py --validate {file_path}")


if __name__ == '__main__':
    main()
