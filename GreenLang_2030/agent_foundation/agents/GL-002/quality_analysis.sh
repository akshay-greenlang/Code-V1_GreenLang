#!/bin/bash

# Count Python files
TOTAL_FILES=$(find . -name "*.py" -type f | wc -l)
TOTAL_LINES=$(find . -name "*.py" -type f -exec wc -l {} + | tail -1 | awk '{print $1}')

# Check imports
echo "=== Import Analysis ==="
grep -r "^import\|^from" --include="*.py" . | wc -l

# Check for potential issues
echo "=== Potential Issues Analysis ==="
echo "Files with no docstrings:"
find . -name "*.py" -type f -exec grep -L '"""' {} \; | wc -l

echo "Files with TODO:"
grep -r "TODO\|FIXME\|XXX" --include="*.py" . | wc -l

echo "Files with bare except:"
grep -r "except:" --include="*.py" . | wc -l

echo "Files with eval/exec:"
grep -r "eval(\|exec(" --include="*.py" . | wc -l

echo "=== Results ==="
echo "Total Python files: $TOTAL_FILES"
echo "Total lines of code: $TOTAL_LINES"

