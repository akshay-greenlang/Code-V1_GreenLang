#!/bin/bash
# Count lines of code for infrastructure modules

echo "Module,Files,Lines,Status"

for module in intelligence sdk cache validation telemetry db auth config provenance services agents/templates; do
    dir="/c/Users/aksha/Code-V1_GreenLang/greenlang/$module"
    if [ -d "$dir" ]; then
        files=$(find "$dir" -name "*.py" 2>/dev/null | wc -l)
        lines=$(find "$dir" -name "*.py" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
        echo "$module,$files,$lines,Complete"
    else
        echo "$module,0,0,Missing"
    fi
done
