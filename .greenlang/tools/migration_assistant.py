#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Migration Assistant Wizard

Step-by-step interactive migration wizard.
Guides users through migrating to infrastructure components.
"""

import argparse
import os
import sys
from typing import Dict, List, Any
import json


class MigrationWizard:
    """Interactive migration wizard."""

    MIGRATION_TYPES = {
        '1': {
            'name': 'LLM Migration',
            'description': 'Migrate from direct OpenAI/Anthropic to ChatSession',
            'from': 'openai.ChatCompletion or anthropic.Client',
            'to': 'shared.infrastructure.llm.ChatSession'
        },
        '2': {
            'name': 'Agent Migration',
            'description': 'Convert custom agents to BaseAgent',
            'from': 'Custom agent classes',
            'to': 'shared.infrastructure.agents.BaseAgent'
        },
        '3': {
            'name': 'Cache Migration',
            'description': 'Replace custom caching with CacheManager',
            'from': 'dict, lru_cache, or custom Redis',
            'to': 'shared.infrastructure.cache.CacheManager'
        },
        '4': {
            'name': 'Logging Migration',
            'description': 'Replace print() with Logger',
            'from': 'print() statements',
            'to': 'shared.infrastructure.logging.Logger'
        },
        '5': {
            'name': 'Config Migration',
            'description': 'Centralize configuration',
            'from': 'os.getenv(), manual config loading',
            'to': 'shared.infrastructure.config.ConfigManager'
        }
    }

    def __init__(self):
        self.migration_type = None
        self.files_to_migrate = []
        self.migration_plan = []

    def start(self):
        """Start the migration wizard."""
        print("=" * 80)
        print("GREENLANG MIGRATION ASSISTANT")
        print("=" * 80)
        print("\nThis wizard will guide you through migrating to GreenLang infrastructure.\n")

        # Step 1: Choose migration type
        self.choose_migration_type()

        # Step 2: Scan for files
        self.scan_files()

        # Step 3: Show migration plan
        self.show_migration_plan()

        # Step 4: Execute migration
        self.execute_migration()

    def choose_migration_type(self):
        """Let user choose migration type."""
        print("Step 1: Choose Migration Type")
        print("-" * 80)

        for key, migration in self.MIGRATION_TYPES.items():
            print(f"\n{key}. {migration['name']}")
            print(f"   {migration['description']}")
            print(f"   From: {migration['from']}")
            print(f"   To:   {migration['to']}")

        choice = input("\nEnter your choice (1-5): ").strip()

        if choice in self.MIGRATION_TYPES:
            self.migration_type = self.MIGRATION_TYPES[choice]
            print(f"\n✓ Selected: {self.migration_type['name']}\n")
        else:
            print("Invalid choice. Exiting.")
            sys.exit(1)

    def scan_files(self):
        """Scan for files that need migration."""
        print("Step 2: Scanning for Files")
        print("-" * 80)

        directory = input("Enter directory to scan (default: current directory): ").strip() or "."

        if not os.path.exists(directory):
            print(f"Directory {directory} not found.")
            sys.exit(1)

        print(f"\nScanning {directory}...\n")

        # Scan based on migration type
        if self.migration_type['name'] == 'LLM Migration':
            self.files_to_migrate = self._scan_for_llm_usage(directory)
        elif self.migration_type['name'] == 'Agent Migration':
            self.files_to_migrate = self._scan_for_agents(directory)
        elif self.migration_type['name'] == 'Cache Migration':
            self.files_to_migrate = self._scan_for_caching(directory)
        elif self.migration_type['name'] == 'Logging Migration':
            self.files_to_migrate = self._scan_for_print(directory)
        elif self.migration_type['name'] == 'Config Migration':
            self.files_to_migrate = self._scan_for_config(directory)

        print(f"✓ Found {len(self.files_to_migrate)} files to migrate\n")

        if self.files_to_migrate:
            print("Files:")
            for i, file_path in enumerate(self.files_to_migrate[:10], 1):
                print(f"  {i}. {file_path}")
            if len(self.files_to_migrate) > 10:
                print(f"  ... and {len(self.files_to_migrate) - 10} more")
        else:
            print("No files found that need migration.")
            sys.exit(0)

    def _scan_for_llm_usage(self, directory: str) -> List[str]:
        """Scan for LLM usage."""
        files = []
        for root, dirs, filenames in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'venv']]
            for filename in filenames:
                if filename.endswith('.py'):
                    file_path = os.path.join(root, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if 'openai.' in content or 'anthropic.' in content:
                                files.append(file_path)
                    except:
                        pass
        return files

    def _scan_for_agents(self, directory: str) -> List[str]:
        """Scan for agent classes."""
        files = []
        for root, dirs, filenames in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'venv']]
            for filename in filenames:
                if filename.endswith('.py') and 'agent' in filename.lower():
                    file_path = os.path.join(root, filename)
                    files.append(file_path)
        return files

    def _scan_for_caching(self, directory: str) -> List[str]:
        """Scan for caching."""
        files = []
        for root, dirs, filenames in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'venv']]
            for filename in filenames:
                if filename.endswith('.py'):
                    file_path = os.path.join(root, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if '@lru_cache' in content or 'redis.Redis' in content:
                                files.append(file_path)
                    except:
                        pass
        return files

    def _scan_for_print(self, directory: str) -> List[str]:
        """Scan for print statements."""
        files = []
        for root, dirs, filenames in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'venv']]
            for filename in filenames:
                if filename.endswith('.py'):
                    file_path = os.path.join(root, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if 'print(' in content and 'logger' not in content.lower():
                                files.append(file_path)
                    except:
                        pass
        return files

    def _scan_for_config(self, directory: str) -> List[str]:
        """Scan for config loading."""
        files = []
        for root, dirs, filenames in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'venv']]
            for filename in filenames:
                if filename.endswith('.py'):
                    file_path = os.path.join(root, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if 'os.getenv' in content or 'os.environ[' in content:
                                files.append(file_path)
                    except:
                        pass
        return files

    def show_migration_plan(self):
        """Show migration plan."""
        print("\nStep 3: Migration Plan")
        print("-" * 80)
        print(f"\nMigration: {self.migration_type['name']}")
        print(f"Files to migrate: {len(self.files_to_migrate)}")

        print("\nSteps:")
        if self.migration_type['name'] == 'LLM Migration':
            print("  1. Add import: from shared.infrastructure.llm import ChatSession")
            print("  2. Replace openai.ChatCompletion with ChatSession")
            print("  3. Update API calls to use ChatSession.chat()")
            print("  4. Test the changes")
        elif self.migration_type['name'] == 'Logging Migration':
            print("  1. Add import: from shared.infrastructure.logging import Logger")
            print("  2. Initialize: logger = Logger(name=__name__)")
            print("  3. Replace print() with logger.info()")
            print("  4. Test the changes")

        print("\nEstimated time: 5-15 minutes per file")

    def execute_migration(self):
        """Execute migration."""
        print("\n\nStep 4: Execute Migration")
        print("-" * 80)

        proceed = input("\nProceed with migration? (y/n): ").strip().lower()

        if proceed != 'y':
            print("Migration cancelled.")
            return

        print("\nMigrating files...\n")

        for i, file_path in enumerate(self.files_to_migrate, 1):
            print(f"[{i}/{len(self.files_to_migrate)}] {file_path}")

            # Show what would be changed
            print("  → Would add infrastructure imports")
            print("  → Would replace anti-patterns")
            print("  ✓ Skipped (dry-run mode)")

        print("\n" + "=" * 80)
        print("DRY RUN COMPLETE")
        print("=" * 80)
        print("\nTo actually apply changes, use:")
        print("  greenlang migrate --auto-fix")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Interactive migration wizard')
    args = parser.parse_args()

    wizard = MigrationWizard()
    wizard.start()


if __name__ == '__main__':
    main()
