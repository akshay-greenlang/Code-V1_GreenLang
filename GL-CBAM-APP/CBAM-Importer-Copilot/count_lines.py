import os

# Count lines in all Python files
total_lines = 0
py_files = []

for root, dirs, files in os.walk('.'):
    # Skip __pycache__ and .pytest_cache directories
    if '__pycache__' in root or '.pytest_cache' in root:
        continue

    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            py_files.append(filepath)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = len(f.readlines())
                    total_lines += lines
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

print(f"Total Python files: {len(py_files)}")
print(f"Total Python lines: {total_lines}")

# Count YAML/JSON files
config_lines = 0
config_files = []

for root, dirs, files in os.walk('.'):
    if '__pycache__' in root:
        continue

    for file in files:
        if file.endswith(('.yaml', '.yml', '.json')):
            filepath = os.path.join(root, file)
            config_files.append(filepath)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = len(f.readlines())
                    config_lines += lines
            except:
                pass

print(f"Total config files (YAML/JSON): {len(config_files)}")
print(f"Total config lines: {config_lines}")
print(f"\nGrand Total Lines: {total_lines + config_lines}")
