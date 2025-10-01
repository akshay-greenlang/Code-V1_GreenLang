#!/bin/bash

# GreenLang Quickstart Examples - Linux/macOS Runner
# This script runs all quickstart examples in sequence

set -e  # Exit on any error

echo "🌍 GreenLang Quickstart Examples Runner"
echo "======================================"
echo ""

# Check if GreenLang is installed
echo "🔍 Checking GreenLang installation..."
if ! command -v gl &> /dev/null; then
    echo "❌ GreenLang CLI not found in PATH"
    echo "💡 Install with: pip install greenlang-cli==0.3.0"
    exit 1
fi

# Verify installation
GL_VERSION=$(gl version 2>/dev/null || echo "unknown")
echo "✅ GreenLang CLI found: $GL_VERSION"

# Check Python environment
echo ""
echo "🐍 Checking Python environment..."
python3 -c "import greenlang; print('✅ GreenLang Python package available')" 2>/dev/null || {
    echo "❌ GreenLang Python package not found"
    echo "💡 Install with: pip install greenlang-cli[analytics]==0.3.0"
    exit 1
}

echo ""
echo "🚀 Running Examples..."
echo "====================="

# Create results directory
mkdir -p results

# Example 1: Hello World
echo ""
echo "📍 Example 1: Hello World Calculation"
echo "--------------------------------------"
python3 hello-world.py || {
    echo "❌ Hello World example failed"
    echo "Check your GreenLang installation and try again"
    exit 1
}

echo ""
echo "✅ Hello World example completed successfully!"

# Example 2: Data Processing
echo ""
echo "📍 Example 2: Portfolio Data Processing"
echo "---------------------------------------"
python3 process-data.py || {
    echo "❌ Data processing example failed"
    echo "Check the error messages above"
    exit 1
}

echo ""
echo "✅ Data processing example completed successfully!"

# Example 3: CLI Usage
echo ""
echo "📍 Example 3: CLI Usage Demonstration"
echo "-------------------------------------"

echo "Testing basic CLI calculation..."
gl calc \
  --fuel-type electricity \
  --consumption 1000 \
  --unit kWh \
  --location "San Francisco" \
  --output results/cli_test_simple.json || {
    echo "❌ Simple CLI test failed"
    exit 1
}

echo "✅ Simple CLI calculation completed"

echo "Testing building-specific calculation..."
gl calc \
  --building-type office \
  --area 2500 \
  --fuels "electricity:50000:kWh,natural_gas:1000:therms" \
  --location "San Francisco" \
  --output results/cli_test_building.json || {
    echo "❌ Building CLI test failed"
    exit 1
}

echo "✅ Building-specific CLI calculation completed"

# Example 4: JSON Input Processing
echo ""
echo "📍 Example 4: JSON File Processing"
echo "----------------------------------"

echo "Processing single building..."
gl calc \
  --input sample-building.json \
  --output results/single_building_result.json || {
    echo "❌ Single building processing failed"
    exit 1
}

echo "✅ Single building processing completed"

# Display results summary
echo ""
echo "📊 RESULTS SUMMARY"
echo "=================="

echo "Generated files:"
ls -la results/ | grep -E '\.(json|csv|txt)$' | while read -r line; do
    echo "  📄 $line"
done

echo ""
echo "🎉 All examples completed successfully!"
echo ""
echo "📚 What's next?"
echo "  • Check the 'results' directory for generated reports"
echo "  • Modify the sample data files with your own building data"
echo "  • Explore more examples in ../tutorials/"
echo "  • Read the documentation: https://greenlang.io/docs"
echo "  • Join our community: https://discord.gg/greenlang"
echo ""
echo "🌱 Start making an impact with GreenLang!"