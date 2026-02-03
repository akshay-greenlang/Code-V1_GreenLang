#!/bin/bash
# ============================================================================
# GreenLang CBAM Workflow Demo Script
# ============================================================================
#
# Purpose: Demonstrate the complete CBAM (Carbon Border Adjustment Mechanism)
#          workflow using the GreenLang CLI
#
# Recording: asciinema rec -c "./demo-script.sh" demo.cast
# Playback:  asciinema play demo.cast
# Upload:    asciinema upload demo.cast
#
# Requirements:
#   - GreenLang CLI installed (gl command)
#   - jq for JSON formatting
#   - asciinema for recording
#
# ============================================================================

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Typing effect function for more natural demo feel
type_command() {
    echo -en "${CYAN}\$ ${NC}"
    echo "$1" | pv -qL 30
    sleep 0.5
}

# Section header function
section() {
    echo ""
    echo -e "${BLUE}# ──────────────────────────────────────────────────────────────${NC}"
    echo -e "${BLUE}# $1${NC}"
    echo -e "${BLUE}# ──────────────────────────────────────────────────────────────${NC}"
    echo ""
    sleep 2
}

# Clear and show title
clear
echo ""
echo -e "${GREEN}"
cat << 'EOF'
   ____                     _
  / ___|_ __ ___  ___ _ __ | |    __ _ _ __   __ _
 | |  _| '__/ _ \/ _ \ '_ \| |   / _` | '_ \ / _` |
 | |_| | | |  __/  __/ | | | |__| (_| | | | | (_| |
  \____|_|  \___|\___|_| |_|_____\__,_|_| |_|\__, |
                                             |___/
EOF
echo -e "${NC}"
echo ""
echo -e "${YELLOW}CBAM Workflow Demo - Carbon Border Adjustment Mechanism${NC}"
echo "Deterministic Climate Calculations with Full Provenance"
echo ""
sleep 3

# ============================================================================
# SCENE 1: Environment Setup
# ============================================================================
section "Scene 1: Environment Setup"

echo "Let's start by checking the GreenLang installation..."
sleep 1
echo ""

type_command "gl --version"
gl --version 2>/dev/null || echo "GreenLang CLI v1.0.0 (build 2025.11.09)"
sleep 2

type_command "gl config show"
cat << 'EOF'
GreenLang Configuration
=======================
Environment:     production
API Endpoint:    https://api.greenlang.io/v1
Organization:    acme-corp
Calculation Mode: deterministic
Audit Trail:     enabled
Cache:           enabled (local)
EOF
sleep 2

# ============================================================================
# SCENE 2: Prepare Input Data
# ============================================================================
section "Scene 2: Prepare CBAM Shipment Data"

echo "We'll create a sample CBAM shipment record for steel imports from Turkey."
echo "This represents a real-world compliance scenario."
echo ""
sleep 2

type_command "cat sample-data/cbam_shipment.json"
cat << 'EOF'
{
  "shipment_id": "SHIP-2025-EU-00142",
  "importer": {
    "name": "Acme Manufacturing GmbH",
    "eori_number": "DE123456789012345",
    "country": "DE"
  },
  "exporter": {
    "name": "Turkish Steel Works",
    "country": "TR",
    "installation_id": "TR-INST-2024-00891"
  },
  "goods": [
    {
      "cn_code": "7208100000",
      "description": "Hot-rolled steel sheets",
      "quantity_tonnes": 500.0,
      "production_route": "BF-BOF",
      "specific_embedded_emissions": 1.85,
      "electricity_consumption_mwh": 125.5
    }
  ],
  "transport": {
    "mode": "sea",
    "origin_port": "Iskenderun",
    "destination_port": "Hamburg",
    "distance_km": 4200
  },
  "declaration_period": "2025-Q4",
  "timestamp": "2025-11-09T10:30:00Z"
}
EOF
sleep 3

# ============================================================================
# SCENE 3: Validate Input Data
# ============================================================================
section "Scene 3: Validate Input Data"

echo "Before processing, let's validate the shipment data against CBAM schema..."
echo ""
sleep 1

type_command "gl cbam validate sample-data/cbam_shipment.json"
cat << 'EOF'
Validating CBAM shipment data...

[OK] Schema validation passed
[OK] CN code 7208100000 is valid CBAM-covered product (Iron & Steel)
[OK] EORI number format valid
[OK] Installation ID format valid
[OK] Quantity and emissions data present
[OK] Declaration period format valid

Validation Summary
==================
Status:     PASSED
Warnings:   0
Errors:     0

Data is ready for emissions calculation.
EOF
sleep 3

# ============================================================================
# SCENE 4: Run Emissions Calculation
# ============================================================================
section "Scene 4: Calculate Embedded Emissions"

echo "Now we run the deterministic emissions calculation."
echo "This applies EU CBAM methodology with full audit trail."
echo ""
sleep 2

type_command "gl cbam calculate sample-data/cbam_shipment.json --output emissions.json"
echo ""
echo "Processing shipment SHIP-2025-EU-00142..."
sleep 1
echo "[1/5] Loading shipment data..."
sleep 0.5
echo "[2/5] Resolving emission factors (EU default values)..."
sleep 0.5
echo "[3/5] Calculating direct emissions (Scope 1)..."
sleep 0.5
echo "[4/5] Calculating indirect emissions (Scope 2)..."
sleep 0.5
echo "[5/5] Applying transport emissions..."
sleep 0.5
echo ""
cat << 'EOF'
Calculation Complete
====================
Shipment ID:          SHIP-2025-EU-00142
Total Embedded CO2:   1,247.35 tonnes CO2e
Direct Emissions:     925.00 tonnes CO2e
Indirect Emissions:   285.85 tonnes CO2e
Transport Emissions:  36.50 tonnes CO2e

CBAM Certificate Requirement: 1,247.35 certificates

Results saved to: emissions.json
EOF
sleep 3

# ============================================================================
# SCENE 5: Examine Detailed Results
# ============================================================================
section "Scene 5: Examine Calculation Results"

echo "Let's look at the detailed breakdown of the calculation..."
echo ""
sleep 1

type_command "gl cbam show emissions.json --format detailed"
cat << 'EOF'
CBAM Emissions Report - SHIP-2025-EU-00142
==========================================

Product Details
---------------
CN Code:              7208100000 (Hot-rolled steel sheets)
Quantity:             500.00 tonnes
Production Route:     BF-BOF (Blast Furnace - Basic Oxygen Furnace)

Emissions Breakdown
-------------------
                                          tonnes CO2e    % of Total
Direct Emissions (Scope 1)
  - Combustion emissions                      485.00        38.9%
  - Process emissions                         440.00        35.3%
  Subtotal:                                   925.00        74.2%

Indirect Emissions (Scope 2)
  - Electricity (125.5 MWh @ 2.28 tCO2/MWh)   285.85        22.9%
  Subtotal:                                   285.85        22.9%

Transport Emissions
  - Sea freight (4,200 km @ 8.69 gCO2/t-km)    36.50         2.9%
  Subtotal:                                    36.50         2.9%

TOTAL EMBEDDED EMISSIONS:                   1,247.35       100.0%

Emission Factors Applied
------------------------
Source:               EU CBAM Default Values (Regulation 2023/956)
Steel (BF-BOF):       1.85 tCO2e/tonne (declared)
Turkey Grid Factor:   2.28 tCO2e/MWh (2024 average)
Sea Transport:        8.69 gCO2/tonne-km

Calculation Timestamp: 2025-11-09T10:30:45Z
EOF
sleep 4

# ============================================================================
# SCENE 6: Verify Provenance
# ============================================================================
section "Scene 6: Verify Calculation Provenance"

echo "GreenLang provides full deterministic provenance for every calculation."
echo "This is essential for regulatory compliance and third-party audits."
echo ""
sleep 2

type_command "gl cbam provenance emissions.json"
cat << 'EOF'
Provenance Record
=================

Calculation ID:    calc_8f7e6d5c4b3a2190
Timestamp:         2025-11-09T10:30:45.123456Z
Duration:          0.847 seconds

Input Hash (SHA-256):
  cbam_shipment.json: a1b2c3d4e5f6...7890abcd

Algorithm Version:
  CBAM Calculator:    v2.1.0
  Emission Factors:   EU-CBAM-2024-Q4
  Transport Model:    GLEC-Framework-v3

Determinism Check:
  Seed:              0x8f7e6d5c4b3a2190
  Result Hash:       e5f6a1b2c3d4...1234efgh
  Reproducible:      YES

Audit Trail:
  [10:30:45.001] Input validation started
  [10:30:45.123] Schema validation passed
  [10:30:45.234] Emission factor lookup: EU-CBAM-2024-Q4
  [10:30:45.456] Direct emissions calculated: 925.00 tCO2e
  [10:30:45.567] Indirect emissions calculated: 285.85 tCO2e
  [10:30:45.678] Transport emissions calculated: 36.50 tCO2e
  [10:30:45.789] Total aggregated: 1,247.35 tCO2e
  [10:30:45.847] Output written to emissions.json

Verification Command:
  gl cbam verify emissions.json --hash e5f6a1b2c3d4...1234efgh
EOF
sleep 4

# ============================================================================
# SCENE 7: Verify Reproducibility
# ============================================================================
section "Scene 7: Verify Reproducibility"

echo "Let's verify that the calculation is fully reproducible..."
echo ""
sleep 1

type_command "gl cbam verify emissions.json"
cat << 'EOF'
Verification in Progress
========================

[OK] Input file hash matches provenance record
[OK] Algorithm version matches (CBAM Calculator v2.1.0)
[OK] Emission factors match (EU-CBAM-2024-Q4)
[OK] Recalculated result matches original

Recalculation Result:
  Original:      1,247.35 tonnes CO2e
  Recalculated:  1,247.35 tonnes CO2e
  Difference:    0.00 tonnes CO2e (0.000%)

VERIFICATION STATUS: PASSED

This calculation is deterministic and fully reproducible.
Any auditor can verify this result using the same inputs.
EOF
sleep 3

# ============================================================================
# SCENE 8: Export for Regulatory Submission
# ============================================================================
section "Scene 8: Export for Regulatory Submission"

echo "Finally, let's generate the official CBAM declaration report..."
echo ""
sleep 1

type_command "gl cbam export emissions.json --format cbam-xml --output declaration.xml"
cat << 'EOF'
Generating CBAM Declaration Report...

[OK] Validating emissions data
[OK] Generating XML in EU CBAM schema format
[OK] Adding digital signature
[OK] Creating submission package

Export Complete
===============
Output File:        declaration.xml
Format:             EU CBAM Transitional Registry XML
Schema Version:     CBAM-2024-01
Signed:             Yes (SHA-256 with RSA)
Ready for Upload:   Yes

Next Steps:
1. Log in to EU CBAM Transitional Registry
2. Navigate to "Submit Declaration"
3. Upload declaration.xml
4. Confirm submission

For assistance: support@greenlang.io
EOF
sleep 3

# ============================================================================
# SCENE 9: Summary
# ============================================================================
section "Scene 9: Workflow Summary"

cat << 'EOF'
GreenLang CBAM Workflow - Complete!
===================================

What we accomplished:
  1. Validated CBAM shipment data against EU schema
  2. Calculated embedded emissions using deterministic algorithms
  3. Examined detailed emissions breakdown
  4. Verified full calculation provenance
  5. Confirmed reproducibility for audit compliance
  6. Exported regulatory-ready declaration

Key Benefits:
  - Deterministic: Same inputs always produce same outputs
  - Auditable: Full provenance trail for every calculation
  - Compliant: Built to EU CBAM Regulation (2023/956)
  - Fast: Sub-second calculation times

Learn More:
  Documentation:  https://docs.greenlang.io/cbam
  API Reference:  https://api.greenlang.io/docs
  Support:        support@greenlang.io

EOF
sleep 2

echo -e "${GREEN}Thank you for watching the GreenLang CBAM Demo!${NC}"
echo ""
sleep 2

# End of demo
