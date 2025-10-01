#!/bin/bash

# GreenLang Quickstart Examples - Docker Runner
# This script runs examples using Docker containers

set -e  # Exit on any error

echo "🐳 GreenLang Docker Quickstart Examples"
echo "======================================="
echo ""

# Check if Docker is available
echo "🔍 Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found in PATH"
    echo "💡 Install Docker from: https://docker.com/get-started"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo "❌ Docker daemon not running"
    echo "💡 Start Docker Desktop or docker daemon"
    exit 1
fi

echo "✅ Docker is available and running"

# Set Docker image
GREENLANG_IMAGE="ghcr.io/greenlang/greenlang:0.3.0"

# Pull the latest image
echo ""
echo "📥 Pulling GreenLang Docker image..."
docker pull $GREENLANG_IMAGE || {
    echo "❌ Failed to pull Docker image"
    echo "💡 Check your internet connection and try again"
    exit 1
}

echo "✅ Docker image pulled successfully"

# Create results directory
mkdir -p results

echo ""
echo "🚀 Running Docker Examples..."
echo "============================="

# Example 1: Version Check
echo ""
echo "📍 Example 1: Version Check"
echo "---------------------------"
docker run --rm $GREENLANG_IMAGE version || {
    echo "❌ Version check failed"
    exit 1
}
echo "✅ Version check completed"

# Example 2: Simple Calculation
echo ""
echo "📍 Example 2: Simple Calculation"
echo "--------------------------------"
docker run --rm \
  -v "$(pwd)/results:/app/results" \
  $GREENLANG_IMAGE \
  calc \
    --fuel-type electricity \
    --consumption 1000 \
    --unit kWh \
    --location "San Francisco" \
    --output /app/results/docker_simple_calc.json || {
    echo "❌ Simple calculation failed"
    exit 1
}
echo "✅ Simple calculation completed"

# Example 3: Building Analysis with JSON Input
echo ""
echo "📍 Example 3: Building Analysis"
echo "-------------------------------"
docker run --rm \
  -v "$(pwd):/app/data" \
  -v "$(pwd)/results:/app/results" \
  $GREENLANG_IMAGE \
  calc \
    --input /app/data/sample-building.json \
    --output /app/results/docker_building_analysis.json || {
    echo "❌ Building analysis failed"
    exit 1
}
echo "✅ Building analysis completed"

# Example 4: Portfolio Analysis
echo ""
echo "📍 Example 4: Portfolio Analysis"
echo "--------------------------------"
docker run --rm \
  -v "$(pwd):/app/data" \
  -v "$(pwd)/results:/app/results" \
  $GREENLANG_IMAGE \
  calc \
    --input /app/data/sample-portfolio.json \
    --output /app/results/docker_portfolio_analysis.json \
    --format detailed || {
    echo "❌ Portfolio analysis failed"
    exit 1
}
echo "✅ Portfolio analysis completed"

# Example 5: Data Validation
echo ""
echo "📍 Example 5: Data Validation"
echo "-----------------------------"
docker run --rm \
  -v "$(pwd):/app/data" \
  $GREENLANG_IMAGE \
  validate-data \
    --input /app/data/sample-building.json \
    --strict || {
    echo "❌ Data validation failed"
    exit 1
}
echo "✅ Data validation completed"

# Example 6: Interactive Python Session (if supported)
echo ""
echo "📍 Example 6: Python SDK in Container"
echo "-------------------------------------"
docker run --rm \
  -v "$(pwd):/app/data" \
  -v "$(pwd)/results:/app/results" \
  $GREENLANG_IMAGE \
  python3 -c "
from greenlang.sdk import GreenLangClient
import json

# Initialize client
client = GreenLangClient()

# Load sample data
with open('/app/data/sample-building.json', 'r') as f:
    building_data = json.load(f)

# Quick calculation
fuels = building_data['energy_consumption']
result = client.calculate_carbon_footprint(fuels)

if result.get('success'):
    print(f'✅ Container calculation: {result[\"data\"][\"total_emissions_tons\"]:.2f} tCO2e')

    # Save result
    with open('/app/results/docker_python_result.json', 'w') as f:
        json.dump(result, f, indent=2)
else:
    print(f'❌ Calculation failed: {result.get(\"errors\")}')
    exit(1)
" || {
    echo "❌ Python SDK test failed"
    exit 1
}
echo "✅ Python SDK test completed"

# Display results
echo ""
echo "📊 DOCKER RESULTS SUMMARY"
echo "========================="

echo "Generated files:"
ls -la results/ | grep -E 'docker.*\.(json|csv|txt)$' | while read -r line; do
    echo "  📄 $line"
done

echo ""
echo "🎉 All Docker examples completed successfully!"
echo ""
echo "📚 Docker-specific tips:"
echo "  • Mount volumes to persist data: -v \$(pwd):/app/data"
echo "  • Use --rm flag to auto-remove containers"
echo "  • Map ports for web interfaces: -p 8080:8080"
echo "  • Set environment variables: -e GL_DEBUG=1"
echo ""
echo "🔗 Next steps:"
echo "  • Compare results with native Python examples"
echo "  • Try running examples in Kubernetes"
echo "  • Set up Docker Compose for complex workflows"
echo "  • Explore container orchestration options"
echo ""
echo "🌱 Ready to deploy GreenLang in production!"