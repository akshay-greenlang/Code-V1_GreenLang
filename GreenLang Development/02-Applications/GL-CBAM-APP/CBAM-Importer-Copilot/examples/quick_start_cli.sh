#!/bin/bash
# ============================================================================
# CBAM IMPORTER COPILOT - CLI QUICK START
# ============================================================================
#
# This script demonstrates how to use the CBAM Copilot CLI commands.
#
# Prerequisites:
#   1. Install dependencies: pip install -r requirements.txt
#   2. Install GreenLang CLI: pip install greenlang-cli>=0.3.0
#
# Usage:
#   bash examples/quick_start_cli.sh
#
# Version: 1.0.0
#
# ============================================================================

set -e  # Exit on error

echo "================================"
echo "CBAM COPILOT - CLI QUICK START"
echo "================================"
echo

# ----------------------------------------------------------------------------
# STEP 1: Create Configuration (One-time setup)
# ----------------------------------------------------------------------------

echo "STEP 1: Creating configuration file..."
echo "--------------------------------------"

# Check if .cbam.yaml already exists
if [ -f ".cbam.yaml" ]; then
    echo "✓ Configuration file already exists: .cbam.yaml"
    echo "  (Delete it to recreate)"
else
    # Create configuration from template
    cp config/cbam_config.yaml .cbam.yaml

    # Update with example values
    cat > .cbam.yaml << 'EOF'
# CBAM Configuration
importer:
  name: "Acme Steel EU BV"
  country: "NL"
  eori: "NL123456789012"

declarant:
  name: "John Smith"
  position: "Compliance Officer"

paths:
  cn_codes: "data/cn_codes.json"
  rules: "rules/cbam_rules.yaml"
  suppliers: "examples/demo_suppliers.yaml"
EOF

    echo "✓ Created .cbam.yaml with example configuration"
fi

echo

# ----------------------------------------------------------------------------
# STEP 2: Validate Shipment Data
# ----------------------------------------------------------------------------

echo "STEP 2: Validating shipment data..."
echo "------------------------------------"

# Using Python CLI directly (GreenLang CLI integration pending)
python cbam_pipeline.py \
  --input examples/demo_shipments.csv \
  --output /dev/null \
  --importer-name "Acme Steel EU BV" \
  --importer-country NL \
  --importer-eori NL123456789012 \
  --declarant-name "John Smith" \
  --declarant-position "Compliance Officer" \
  > /tmp/cbam_validation.log 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Shipment data is valid"
else
    echo "✗ Validation failed - check /tmp/cbam_validation.log"
    exit 1
fi

echo

# ----------------------------------------------------------------------------
# STEP 3: Generate CBAM Report (Basic)
# ----------------------------------------------------------------------------

echo "STEP 3: Generating CBAM report (basic)..."
echo "------------------------------------------"

# Create output directory
mkdir -p output

# Generate report using pipeline directly
python cbam_pipeline.py \
  --input examples/demo_shipments.csv \
  --output output/cbam_report_basic.json \
  --importer-name "Acme Steel EU BV" \
  --importer-country NL \
  --importer-eori NL123456789012 \
  --declarant-name "John Smith" \
  --declarant-position "Compliance Officer"

if [ $? -eq 0 ]; then
    echo "✓ Report generated: output/cbam_report_basic.json"
else
    echo "✗ Report generation failed"
    exit 1
fi

echo

# ----------------------------------------------------------------------------
# STEP 4: Generate Complete Report with All Outputs
# ----------------------------------------------------------------------------

echo "STEP 4: Generating complete report with all outputs..."
echo "-------------------------------------------------------"

# Generate with summary and intermediate outputs
python cbam_pipeline.py \
  --input examples/demo_shipments.csv \
  --output output/cbam_report_complete.json \
  --summary output/cbam_summary_complete.md \
  --intermediate output/intermediate \
  --importer-name "Acme Steel EU BV" \
  --importer-country NL \
  --importer-eori NL123456789012 \
  --declarant-name "John Smith" \
  --declarant-position "Compliance Officer"

if [ $? -eq 0 ]; then
    echo "✓ Complete report generated with all outputs:"
    echo "  - JSON report: output/cbam_report_complete.json"
    echo "  - Markdown summary: output/cbam_summary_complete.md"
    echo "  - Intermediate outputs: output/intermediate/"
else
    echo "✗ Report generation failed"
    exit 1
fi

echo

# ----------------------------------------------------------------------------
# STEP 5: View Report Summary
# ----------------------------------------------------------------------------

echo "STEP 5: Viewing report summary..."
echo "----------------------------------"

if [ -f "output/cbam_summary_complete.md" ]; then
    echo
    cat output/cbam_summary_complete.md | head -n 30
    echo
    echo "[... see output/cbam_summary_complete.md for full summary ...]"
else
    echo "⚠ Summary file not found"
fi

echo

# ----------------------------------------------------------------------------
# STEP 6: Extract Key Metrics from JSON Report
# ----------------------------------------------------------------------------

echo "STEP 6: Extracting key metrics from JSON report..."
echo "---------------------------------------------------"

if command -v jq &> /dev/null; then
    echo "Using jq to extract metrics:"
    echo

    echo "Report ID:"
    jq -r '.report_metadata.report_id' output/cbam_report_complete.json

    echo
    echo "Total Emissions:"
    jq -r '.emissions_summary.total_embedded_emissions_tco2' output/cbam_report_complete.json
    echo "tCO2"

    echo
    echo "Total Shipments:"
    jq -r '.goods_summary.total_shipments' output/cbam_report_complete.json

    echo
    echo "Validation Status:"
    jq -r '.validation_results.is_valid' output/cbam_report_complete.json
else
    echo "⚠ jq not installed - install with: sudo apt-get install jq"
    echo "  (or view JSON file directly)"
fi

echo

# ----------------------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------------------

echo "========================================"
echo "QUICK START COMPLETE! 🎉"
echo "========================================"
echo
echo "What you've learned:"
echo "  ✓ How to create a configuration file"
echo "  ✓ How to validate shipment data"
echo "  ✓ How to generate CBAM reports"
echo "  ✓ How to work with report outputs"
echo
echo "Next steps:"
echo "  1. Customize .cbam.yaml with your company info"
echo "  2. Prepare your actual shipment data (CSV/JSON/Excel)"
echo "  3. Run: python cbam_pipeline.py --input YOUR_DATA.csv --output YOUR_REPORT.json"
echo "  4. Review the generated report and summary"
echo "  5. Submit to EU CBAM Transitional Registry"
echo
echo "For help:"
echo "  - Read README.md for full documentation"
echo "  - Check examples/ for more examples"
echo "  - Email: cbam@greenlang.io"
echo
echo "========================================"

# ============================================================================
# END OF QUICK START
# ============================================================================
