# -*- coding: utf-8 -*-
"""
Script to replace all 'greenlang' CLI commands with 'gl' 
"""

import os
import re
from pathlib import Path

def replace_in_file(filepath, replacements):
    """Replace patterns in a file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        for old, new in replacements:
            content = re.sub(old, new, content)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


# Define replacements - using regex to ensure we only replace CLI commands
replacements = [
    # Basic commands
    (r'greenlang\s+calc', 'gl calc'),
    (r'greenlang\s+analyze', 'gl analyze'),
    (r'greenlang\s+benchmark', 'gl benchmark'),
    (r'greenlang\s+recommend', 'gl recommend'),
    (r'greenlang\s+agents', 'gl agents'),
    (r'greenlang\s+ask', 'gl ask'),
    (r'greenlang\s+run', 'gl run'),
    (r'greenlang\s+init', 'gl init'),
    (r'greenlang\s+dev', 'gl dev'),
    (r'greenlang\s+test', 'gl test'),
    (r'greenlang\s+workflow', 'gl workflow'),
    
    # Commands with flags
    (r'greenlang\s+--help', 'gl --help'),
    (r'greenlang\s+--version', 'gl --version'),
    
    # In quotes (for examples in documentation)
    (r'"greenlang\s+', '"gl '),
    (r"'greenlang\s+", "'gl "),
    (r'`greenlang\s+', '`gl '),
    
    # Commands at start of line
    (r'^greenlang\s+', 'gl '),
]

# Files to process
file_patterns = [
    '*.py',
    '*.md',
    '*.txt',
    '*.yaml',
    '*.yml',
    '*.json',
    '*.toml',
    'README*',
]

# Directories to skip
skip_dirs = {
    '.git',
    '.venv',
    '__pycache__',
    'node_modules',
    '.pytest_cache',
    'build',
    'dist',
}

def process_files():
    """Process all files"""
    root = Path('.')
    total_files = 0
    modified_files = 0
    
    for pattern in file_patterns:
        for filepath in root.rglob(pattern):
            # Skip directories we don't want to process
            if any(skip_dir in str(filepath) for skip_dir in skip_dirs):
                continue
            
            # Skip the new gl CLI files themselves
            if 'core/greenlang/cli' in str(filepath):
                continue
            
            # Skip this script
            if filepath.name == 'replace_greenlang_with_gl.py':
                continue
            
            total_files += 1
            if replace_in_file(filepath, replacements):
                modified_files += 1
                print(f"Modified: {filepath}")
    
    print(f"\nProcessed {total_files} files, modified {modified_files} files")


if __name__ == '__main__':
    print("Replacing 'greenlang' CLI commands with 'gl'...")
    print("=" * 50)
    process_files()
    print("=" * 50)
    print("Done! All 'greenlang' commands have been replaced with 'gl'")