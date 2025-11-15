#!/bin/bash

echo "=== CRITICAL ISSUE ANALYSIS ==="
echo ""

echo "1. IMPORT ISSUES:"
echo "---"
for file in $(find . -name "*.py" -type f); do
    # Check for relative imports that might fail
    if grep -q "^from provenance import" "$file" 2>/dev/null; then
        echo "WARNING: Relative import in $file (should be from .provenance)"
    fi
    # Check for unresolved imports
    if grep -q "from agent_foundation" "$file" 2>/dev/null; then
        # This is OK for main orchestrator
        :
    fi
done

echo ""
echo "2. TYPE HINTS COVERAGE:"
echo "---"
# Check for functions without type hints
grep -r "def " --include="*.py" . | grep -v " -> " | wc -l | xargs echo "Functions without return type hints:"
grep -r "def " --include="*.py" . | grep -E "def \w+\([^:]*\):" | wc -l | xargs echo "Functions without parameter type hints:"

echo ""
echo "3. DOCSTRING ANALYSIS:"
echo "---"
# Check class docstrings
grep -r "^class " --include="*.py" . | wc -l | xargs echo "Total classes:"
grep -rB1 "^class " --include="*.py" . | grep '"""' | wc -l | xargs echo "Classes with docstrings:"

echo ""
echo "4. COMPLEXITY INDICATORS:"
echo "---"
# Check for long functions (more than 50 lines might be complex)
echo "Files to review for complexity:"
find . -name "*.py" -type f -exec sh -c 'lines=$(wc -l < "$1"); if [ "$lines" -gt 500 ]; then echo "$1 ($lines lines)"; fi' _ {} \;

echo ""
echo "5. ERROR HANDLING:"
echo "---"
echo "Except blocks found:"
grep -r "except " --include="*.py" . | wc -l
echo "Exception types specified:"
grep -r "except.*:" --include="*.py" . | grep -v "except.*Error\|except.*Exception" | wc -l | xargs echo "Bare/generic exceptions:"

echo ""
echo "6. SECURITY SCAN:"
echo "---"
echo "Hardcoded credentials/secrets:"
grep -r "password\|secret\|api_key\|token" --include="*.py" . | grep -i "=\s*['\"]" | wc -l
echo "Dangerous functions:"
grep -r "subprocess\|os.system\|shell=True" --include="*.py" . | wc -l

echo ""
echo "7. LOGGING ANALYSIS:"
echo "---"
echo "Logger instances:"
grep -r "logger = " --include="*.py" . | wc -l
echo "Logger usage:"
grep -r "logger\." --include="*.py" . | wc -l

echo ""
echo "8. DATACLASS ANALYSIS:"
echo "---"
echo "@dataclass decorators found:"
grep -r "@dataclass" --include="*.py" . | wc -l

echo ""
echo "9. ASYNC/AWAIT USAGE:"
echo "---"
echo "Async functions:"
grep -r "async def" --include="*.py" . | wc -l
echo "Await calls:"
grep -r "await " --include="*.py" . | wc -l

