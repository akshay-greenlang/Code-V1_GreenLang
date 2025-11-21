#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Migration script for upgrading pack.yaml to v1.0 specification

Usage:
    python scripts/migrate_pack_yaml_v1.py [pack.yaml]
    python scripts/migrate_pack_yaml_v1.py --directory ./packs
    python scripts/migrate_pack_yaml_v1.py --check pack.yaml
"""

import argparse
import sys
import yaml
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import difflib


def load_manifest(file_path: Path) -> Dict[str, Any]:
    """Load manifest from YAML or JSON file"""
    content = file_path.read_text()
    
    if file_path.suffix in ['.yaml', '.yml']:
        return yaml.safe_load(content)
    elif file_path.suffix == '.json':
        return json.loads(content)
    else:
        # Try YAML first, then JSON
        try:
            return yaml.safe_load(content)
        except:
            return json.loads(content)


def migrate_manifest(data: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
    """
    Migrate pre-v1.0 manifest to v1.0 specification
    
    Returns:
        Migrated manifest dictionary
    """
    migrated = {}
    changes = []
    
    # 1. Handle name field
    if 'name' in data:
        name = data['name'].lower().replace('_', '-')
        if name != data['name']:
            changes.append(f"Normalized name: '{data['name']}' -> '{name}'")
        migrated['name'] = name
    else:
        # Try to infer from directory name
        name = file_path.parent.name.lower().replace('_', '-')
        migrated['name'] = name
        changes.append(f"Added missing name: '{name}'")
    
    # 2. Handle version field
    if 'version' in data:
        version = data['version']
        # Remove 'v' prefix if present
        if version.startswith('v'):
            version = version[1:]
            changes.append(f"Removed 'v' prefix from version: '{data['version']}' -> '{version}'")
        migrated['version'] = version
    else:
        migrated['version'] = "1.0.0"
        changes.append("Added missing version: '1.0.0'")
    
    # 3. Handle kind field (might be 'type' in old format)
    if 'kind' in data:
        migrated['kind'] = data['kind']
    elif 'type' in data:
        migrated['kind'] = data['type']
        changes.append(f"Renamed 'type' to 'kind': '{data['type']}'")
    else:
        migrated['kind'] = "pack"
        changes.append("Added missing kind: 'pack'")
    
    # 4. Handle license field
    if 'license' in data:
        migrated['license'] = data['license']
    else:
        migrated['license'] = "MIT"
        changes.append("Added missing license: 'MIT'")
    
    # 5. Handle contents field (REQUIRED in v1.0)
    if 'contents' in data:
        contents = data['contents']
        # Ensure pipelines array exists
        if 'pipelines' not in contents or not contents['pipelines']:
            # Check if gl.yaml exists
            gl_yaml = file_path.parent / 'gl.yaml'
            if gl_yaml.exists():
                contents['pipelines'] = ['gl.yaml']
                changes.append("Added 'gl.yaml' to contents.pipelines")
            else:
                contents['pipelines'] = ['pipeline.yaml']
                changes.append("Added default pipeline to contents.pipelines")
        migrated['contents'] = contents
    else:
        # Create contents with gl.yaml if it exists
        contents = {}
        gl_yaml = file_path.parent / 'gl.yaml'
        if gl_yaml.exists():
            contents['pipelines'] = ['gl.yaml']
            changes.append("Created contents with existing gl.yaml")
        else:
            contents['pipelines'] = ['pipeline.yaml']
            changes.append("Created contents with default pipeline.yaml")
        
        # Check for other content types
        if (file_path.parent / 'agents').exists():
            agents = [f.stem for f in (file_path.parent / 'agents').glob('*.py')]
            if agents:
                contents['agents'] = agents
                changes.append(f"Added {len(agents)} agents to contents")
        
        if (file_path.parent / 'datasets').exists():
            datasets = [str(f.relative_to(file_path.parent)) 
                       for f in (file_path.parent / 'datasets').glob('*')
                       if f.is_file()]
            if datasets:
                contents['datasets'] = datasets
                changes.append(f"Added {len(datasets)} datasets to contents")
        
        migrated['contents'] = contents
    
    # 6. Handle compatibility (rename old fields)
    compat = {}
    if 'greenlang_version' in data:
        compat['greenlang'] = data['greenlang_version']
        changes.append("Moved 'greenlang_version' to 'compat.greenlang'")
    elif 'min_greenlang_version' in data:
        compat['greenlang'] = f">={data['min_greenlang_version']}"
        changes.append("Converted 'min_greenlang_version' to 'compat.greenlang'")
    
    if 'python_version' in data:
        compat['python'] = data['python_version']
        changes.append("Moved 'python_version' to 'compat.python'")
    elif 'min_python_version' in data:
        compat['python'] = f">={data['min_python_version']}"
        changes.append("Converted 'min_python_version' to 'compat.python'")
    
    if compat:
        migrated['compat'] = compat
    
    # 7. Handle dependencies
    if 'dependencies' in data:
        migrated['dependencies'] = data['dependencies']
    elif 'requirements' in data:
        migrated['dependencies'] = data['requirements']
        changes.append("Renamed 'requirements' to 'dependencies'")
    
    # 8. Handle card/documentation
    if 'card' in data:
        migrated['card'] = data['card']
    elif 'readme' in data:
        migrated['card'] = data['readme']
        changes.append("Renamed 'readme' to 'card'")
    elif (file_path.parent / 'CARD.md').exists():
        migrated['card'] = 'CARD.md'
        changes.append("Added reference to existing CARD.md")
    elif (file_path.parent / 'README.md').exists():
        migrated['card'] = 'README.md'
        changes.append("Added reference to existing README.md")
    
    # 9. Copy over other v1.0 compatible fields
    for field in ['policy', 'security', 'metadata']:
        if field in data:
            migrated[field] = data[field]
    
    # 10. Handle old metadata fields
    if 'description' in data and 'metadata' not in migrated:
        migrated['metadata'] = {'description': data['description']}
        changes.append("Moved 'description' to 'metadata.description'")
    elif 'description' in data and 'metadata' in migrated:
        migrated['metadata']['description'] = data['description']
        changes.append("Moved 'description' to 'metadata.description'")
    
    if 'author' in data or 'authors' in data:
        if 'metadata' not in migrated:
            migrated['metadata'] = {}
        authors = data.get('authors', [data.get('author')] if 'author' in data else [])
        if authors:
            migrated['metadata']['authors'] = authors if isinstance(authors, list) else [authors]
            changes.append("Moved author information to 'metadata.authors'")
    
    # Print migration summary
    if changes:
        print(f"\n[Migration changes] for {file_path}:")
        for change in changes:
            print(f"   - {change}")
    
    return migrated


def validate_migrated(data: Dict[str, Any]) -> List[str]:
    """Validate migrated manifest against v1.0 spec"""
    errors = []
    warnings = []
    
    # Check required fields
    required = ['name', 'version', 'kind', 'license', 'contents']
    for field in required:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    # Check contents.pipelines
    if 'contents' in data:
        if 'pipelines' not in data['contents']:
            errors.append("Missing required field: contents.pipelines")
        elif not data['contents']['pipelines']:
            errors.append("contents.pipelines must have at least one pipeline")
    
    # Check name format
    if 'name' in data:
        import re
        if not re.match(r'^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$', data['name']):
            errors.append(f"Invalid name format: {data['name']}")
    
    # Check version format
    if 'version' in data:
        import re
        if not re.match(r'^\d+\.\d+\.\d+$', data['version']):
            errors.append(f"Invalid version format: {data['version']}")
    
    # Check kind enum
    if 'kind' in data and data['kind'] not in ['pack', 'dataset', 'connector']:
        errors.append(f"Invalid kind: {data['kind']}")
    
    # Warnings for recommended fields
    if 'card' not in data:
        warnings.append("Recommended: Add 'card' field for documentation")
    
    if 'compat' not in data:
        warnings.append("Recommended: Add 'compat' field for version requirements")
    
    return errors, warnings


def show_diff(original: str, migrated: str, file_path: Path):
    """Show diff between original and migrated content"""
    print(f"\n[DIFF] Changes for {file_path}:")
    print("=" * 60)
    
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        migrated.splitlines(keepends=True),
        fromfile=f"{file_path} (original)",
        tofile=f"{file_path} (migrated)",
        lineterm=''
    )
    
    for line in diff:
        if line.startswith('+'):
            print(f"\033[92m{line}\033[0m", end='')  # Green
        elif line.startswith('-'):
            print(f"\033[91m{line}\033[0m", end='')  # Red
        elif line.startswith('@'):
            print(f"\033[96m{line}\033[0m", end='')  # Cyan
        else:
            print(line, end='')
    
    print("\n" + "=" * 60)


def migrate_file(file_path: Path, check_only: bool = False, show_diff_flag: bool = True) -> bool:
    """
    Migrate a single pack.yaml file
    
    Returns:
        True if migration successful, False otherwise
    """
    print(f"\n[Processing] {file_path}")
    
    try:
        # Load current manifest
        data = load_manifest(file_path)
        original_content = file_path.read_text()
        
        # Migrate to v1.0
        migrated = migrate_manifest(data, file_path)
        
        # Validate
        errors, warnings = validate_migrated(migrated)
        
        if errors:
            print(f"[ERROR] Validation errors after migration:")
            for error in errors:
                print(f"   - {error}")
            return False
        
        if warnings:
            print(f"[WARNING] Warnings:")
            for warning in warnings:
                print(f"   - {warning}")
        
        # Generate migrated content
        if file_path.suffix in ['.yaml', '.yml']:
            migrated_content = yaml.dump(migrated, default_flow_style=False, sort_keys=False)
        else:
            migrated_content = json.dumps(migrated, indent=2)
        
        # Show diff
        if show_diff_flag and original_content != migrated_content:
            show_diff(original_content, migrated_content, file_path)
        
        if check_only:
            print(f"[SUCCESS] Migration check completed (no files modified)")
        else:
            # Backup original
            backup_path = file_path.with_suffix(file_path.suffix + '.bak')
            shutil.copy2(file_path, backup_path)
            print(f"[BACKUP] Saved to: {backup_path}")
            
            # Write migrated content
            file_path.write_text(migrated_content)
            print(f"[SUCCESS] Migrated: {file_path}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to migrate {file_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Migrate pack.yaml files to v1.0 specification"
    )
    parser.add_argument(
        'path',
        nargs='?',
        default='pack.yaml',
        help='Path to pack.yaml file or directory (default: pack.yaml)'
    )
    parser.add_argument(
        '--directory', '-d',
        help='Migrate all pack.yaml files in directory'
    )
    parser.add_argument(
        '--check', '-c',
        action='store_true',
        help='Check only, do not modify files'
    )
    parser.add_argument(
        '--no-diff',
        action='store_true',
        help='Do not show diff output'
    )
    
    args = parser.parse_args()
    
    files_to_migrate = []
    
    if args.directory:
        # Find all pack.yaml files in directory
        directory = Path(args.directory)
        files_to_migrate = list(directory.glob('**/pack.yaml'))
        files_to_migrate.extend(directory.glob('**/pack.yml'))
        print(f"Found {len(files_to_migrate)} pack files in {directory}")
    else:
        # Single file
        path = Path(args.path)
        if path.is_dir():
            # Look for pack.yaml in directory
            for name in ['pack.yaml', 'pack.yml']:
                candidate = path / name
                if candidate.exists():
                    files_to_migrate = [candidate]
                    break
        else:
            files_to_migrate = [path]
    
    if not files_to_migrate:
        print("[ERROR] No pack.yaml files found")
        sys.exit(1)
    
    # Migrate files
    success_count = 0
    for file_path in files_to_migrate:
        if migrate_file(file_path, args.check, not args.no_diff):
            success_count += 1
    
    # Summary
    print(f"\n[Summary] Migration Results:")
    print(f"   - Total files: {len(files_to_migrate)}")
    print(f"   - Successful: {success_count}")
    print(f"   - Failed: {len(files_to_migrate) - success_count}")
    
    if success_count < len(files_to_migrate):
        sys.exit(1)


if __name__ == "__main__":
    main()