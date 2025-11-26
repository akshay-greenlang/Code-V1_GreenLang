# GL-010 EMISSIONWATCH Troubleshooting Guide

## Document Control

| Property | Value |
|----------|-------|
| Document ID | GL-010-RUNBOOK-TS-001 |
| Version | 1.0.0 |
| Last Updated | 2025-11-26 |
| Owner | GL-010 Operations Team |
| Classification | Internal |
| Review Cycle | Quarterly |

---

## Table of Contents

1. [Overview](#1-overview)
2. [Troubleshooting Methodology](#2-troubleshooting-methodology)
3. [Emissions Calculation Issues](#3-emissions-calculation-issues)
4. [CEMS Integration Issues](#4-cems-integration-issues)
5. [Compliance Checking Issues](#5-compliance-checking-issues)
6. [Reporting Issues](#6-reporting-issues)
7. [Diagnostic Commands Reference](#7-diagnostic-commands-reference)
8. [Common Error Messages](#8-common-error-messages)
9. [Log Analysis](#9-log-analysis)
10. [Database Diagnostics](#10-database-diagnostics)
11. [Performance Troubleshooting](#11-performance-troubleshooting)
12. [Appendices](#12-appendices)

---

## 1. Overview

### 1.1 Purpose

This Troubleshooting Guide provides comprehensive diagnostic procedures for identifying and resolving issues with the GL-010 EMISSIONWATCH system. It covers emissions calculations, CEMS integration, compliance checking, and reporting functions.

### 1.2 Scope

This guide addresses:
- Emissions calculation discrepancies and errors
- CEMS data acquisition and quality issues
- Compliance determination problems
- Regulatory report generation and submission issues
- System performance and connectivity problems

### 1.3 Prerequisites

Before using this guide, ensure you have:
- Access to GL-010 system commands (`greenlang` CLI)
- Kubernetes cluster access (`kubectl`)
- Database query permissions
- CEMS system access (read-only minimum)
- Log aggregation system access

### 1.4 Support Escalation

If troubleshooting does not resolve the issue:

| Issue Type | Escalation Path | Contact |
|------------|-----------------|---------|
| Emissions Calculation | Emissions Engineering Team | emissions-eng@greenlang.io |
| CEMS Integration | CEMS Integration Team | cems-team@greenlang.io |
| Compliance Logic | Regulatory Compliance Team | compliance@greenlang.io |
| System/Infrastructure | Platform Engineering | platform-eng@greenlang.io |

---

## 2. Troubleshooting Methodology

### 2.1 Standard Troubleshooting Process

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         TROUBLESHOOTING WORKFLOW                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │ IDENTIFY │───►│ ISOLATE  │───►│ DIAGNOSE │───►│ RESOLVE  │───►│ VERIFY   │      │
│  │          │    │          │    │          │    │          │    │          │      │
│  │- Symptoms│    │- Scope   │    │- Root    │    │- Apply   │    │- Test    │      │
│  │- Impact  │    │- Time    │    │  Cause   │    │  Fix     │    │- Monitor │      │
│  │- History │    │- Pattern │    │- Confirm │    │- Document│    │- Close   │      │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘      │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Initial Diagnostic Checklist

Before diving into specific troubleshooting, perform these general checks:

```bash
# 1. Check overall system health
greenlang health --agent GL-010 --full

# 2. Check for active alerts
greenlang alerts list --agent GL-010 --status open

# 3. Check recent deployments
greenlang deployment history --agent GL-010 --last 7d

# 4. Check resource utilization
greenlang metrics summary --agent GL-010 --period last-1h

# 5. Check external dependencies
greenlang dependencies status --agent GL-010

# 6. Review recent changes
greenlang changelog --agent GL-010 --last 7d
```

### 2.3 Information Gathering Template

When troubleshooting, collect the following information:

```markdown
## Issue Information

**Issue Description:**
[Describe what is happening]

**Expected Behavior:**
[Describe what should happen]

**Environment:**
- Facility ID:
- Unit ID:
- Pollutant(s):
- Time Range:

**Symptoms:**
- Error messages:
- Affected functions:
- User impact:

**Timeline:**
- When did issue start:
- Any recent changes:
- Is issue intermittent or constant:

**Steps to Reproduce:**
1.
2.
3.

**Diagnostic Data Collected:**
- [ ] System logs
- [ ] CEMS data export
- [ ] Calculation trace
- [ ] Configuration snapshot
```

---

## 3. Emissions Calculation Issues

### 3.1 TS-EC-001: NOx Calculation Mismatch with CEMS

#### Symptoms
- GL-010 calculated NOx value differs from CEMS reported value
- Discrepancy exceeds acceptable tolerance (typically >2%)
- Regulatory reports show different values than CEMS historian

#### Diagnostic Steps

**Step 1: Verify Raw CEMS Data**

```bash
# Get raw CEMS NOx data
greenlang cems raw-data \
  --facility facility-001 \
  --unit boiler-001 \
  --pollutant NOx \
  --time-range "2025-11-26T10:00:00Z/2025-11-26T11:00:00Z" \
  --output-format detailed

# Compare with GL-010 received data
greenlang data compare \
  --source cems-raw \
  --target gl010-received \
  --facility facility-001 \
  --pollutant NOx \
  --time-range "2025-11-26T10:00:00Z/2025-11-26T11:00:00Z"
```

**Step 2: Check Data Transformation**

```bash
# View transformation applied
greenlang transformation show \
  --facility facility-001 \
  --pollutant NOx \
  --include-formula

# Trace calculation
greenlang emissions trace \
  --facility facility-001 \
  --unit boiler-001 \
  --pollutant NOx \
  --timestamp "2025-11-26T10:30:00Z" \
  --verbose
```

**Step 3: Check Unit Conversions**

```bash
# Verify unit configuration
greenlang config show \
  --facility facility-001 \
  --section "units.NOx"

# Check conversion factors
greenlang emissions conversion-factors \
  --pollutant NOx \
  --from-unit ppm \
  --to-unit "lb/MMBtu"
```

**Step 4: Verify Correction Factors**

```bash
# Check O2 correction applied
greenlang emissions o2-correction \
  --facility facility-001 \
  --unit boiler-001 \
  --time-range "2025-11-26T10:00:00Z/2025-11-26T11:00:00Z"

# Check temperature/pressure correction
greenlang emissions tp-correction \
  --facility facility-001 \
  --unit boiler-001 \
  --time-range "2025-11-26T10:00:00Z/2025-11-26T11:00:00Z"
```

#### Common Causes and Solutions

| Cause | Symptoms | Solution |
|-------|----------|----------|
| O2 correction mismatch | Values differ by consistent % | Verify O2 reference value matches permit |
| Wrong measurement basis | Large consistent offset | Check dry vs. wet basis configuration |
| Unit conversion error | Values off by factor (e.g., 10x) | Verify unit configuration in both systems |
| Averaging period mismatch | Periodic discrepancy | Align averaging periods |
| Data timestamp offset | Shifted correlation | Check timezone and clock sync |
| Diluent basis mismatch | Systematic error | Verify O2 vs. CO2 diluent setting |

**Resolution Example:**

```bash
# If O2 reference mismatch found:
greenlang config update \
  --facility facility-001 \
  --path "emissions.NOx.o2_reference" \
  --value 3.0 \
  --reason "Match permit O2 reference of 3.0%"

# Recalculate affected period
greenlang emissions recalculate \
  --facility facility-001 \
  --pollutant NOx \
  --time-range "2025-11-26T00:00:00Z/2025-11-26T23:59:59Z" \
  --reason "O2 reference correction"
```

---

### 3.2 TS-EC-002: SOx Calculation Not Matching Fuel Sulfur

#### Symptoms
- Calculated SO2 emissions do not correlate with fuel sulfur content
- Mass balance check fails
- SO2 values higher or lower than expected based on fuel analysis

#### Diagnostic Steps

**Step 1: Verify Fuel Data**

```bash
# Get fuel analysis data
greenlang fuel analysis \
  --facility facility-001 \
  --fuel-type coal \
  --time-range "last-30d" \
  --parameters "sulfur,heating_value,moisture,ash"

# Check fuel usage data
greenlang fuel consumption \
  --facility facility-001 \
  --unit boiler-001 \
  --time-range "2025-11-26T00:00:00Z/2025-11-26T23:59:59Z"
```

**Step 2: Check Sulfur Retention**

```bash
# Verify sulfur retention factor
greenlang config show \
  --facility facility-001 \
  --section "emissions.SO2.sulfur_retention"

# Check ash sulfur content
greenlang fuel ash-sulfur \
  --facility facility-001 \
  --time-range "last-30d"
```

**Step 3: Verify Calculation Methodology**

```bash
# Show SO2 calculation method
greenlang emissions methodology \
  --facility facility-001 \
  --pollutant SO2 \
  --show-formula

# Trace specific calculation
greenlang emissions trace \
  --facility facility-001 \
  --unit boiler-001 \
  --pollutant SO2 \
  --timestamp "2025-11-26T10:00:00Z" \
  --show-inputs
```

**Step 4: Mass Balance Check**

```bash
# Perform mass balance calculation
greenlang emissions mass-balance \
  --facility facility-001 \
  --pollutant SO2 \
  --time-range "2025-11-26T00:00:00Z/2025-11-26T23:59:59Z" \
  --fuel-input \
  --cems-output \
  --show-discrepancy
```

#### Common Causes and Solutions

| Cause | Symptoms | Solution |
|-------|----------|----------|
| Outdated fuel analysis | Trending mismatch | Update fuel analysis data |
| Incorrect sulfur retention | Consistent offset | Calibrate retention factor |
| Heating value error | Proportional error | Verify HHV/LHV basis |
| Moisture basis mismatch | Variable error | Check as-received vs. dry basis |
| Multiple fuel blend | Variable error | Verify fuel blending calculations |
| FGD efficiency change | Sudden change | Update scrubber efficiency factor |

---

### 3.3 TS-EC-003: CO2 Emissions Discrepancy

#### Symptoms
- CO2 emissions calculation differs from expected value
- Annual totals do not match fuel-based estimates
- GHG inventory reconciliation fails

#### Diagnostic Steps

**Step 1: Verify Emission Factors**

```bash
# Check CO2 emission factors
greenlang emissions emission-factors \
  --facility facility-001 \
  --pollutant CO2 \
  --all-fuels

# Compare with EPA factors
greenlang emissions compare-factors \
  --facility facility-001 \
  --pollutant CO2 \
  --reference "EPA-AP42" \
  --reference "40-CFR-98"
```

**Step 2: Check Carbon Content**

```bash
# Verify fuel carbon content
greenlang fuel carbon-content \
  --facility facility-001 \
  --all-fuels \
  --time-range "last-12m"

# Check oxidation factor
greenlang config show \
  --facility facility-001 \
  --section "emissions.CO2.oxidation_factor"
```

**Step 3: Verify Calculation Method**

```bash
# Show CO2 calculation methodology
greenlang emissions methodology \
  --facility facility-001 \
  --pollutant CO2 \
  --program "EPA-GHGRP"

# Trace calculation for specific period
greenlang emissions trace \
  --facility facility-001 \
  --pollutant CO2 \
  --time-range "2025-01-01T00:00:00Z/2025-12-31T23:59:59Z" \
  --by-source
```

**Step 4: Compare with Alternative Calculation**

```bash
# Calculate using alternative method
greenlang emissions calculate \
  --facility facility-001 \
  --pollutant CO2 \
  --method "fuel-based" \
  --time-range "2025-01-01T00:00:00Z/2025-12-31T23:59:59Z"

# Calculate using CEMS method
greenlang emissions calculate \
  --facility facility-001 \
  --pollutant CO2 \
  --method "cems" \
  --time-range "2025-01-01T00:00:00Z/2025-12-31T23:59:59Z"

# Compare results
greenlang emissions compare-methods \
  --facility facility-001 \
  --pollutant CO2 \
  --methods "fuel-based,cems" \
  --time-range "2025-01-01T00:00:00Z/2025-12-31T23:59:59Z"
```

#### Common Causes and Solutions

| Cause | Symptoms | Solution |
|-------|----------|----------|
| Wrong emission factor | Consistent offset | Use correct fuel-specific factor |
| Incomplete fuel tracking | Underestimate | Add missing fuel streams |
| Oxidation factor error | Small consistent error | Verify oxidation factor |
| Biogenic carbon exclusion | Overestimate | Apply biogenic adjustment |
| CEMS CO2 vs calculated | Method difference | Align methodology with regulation |
| Unit conversion | Factor of error | Verify MT vs. tons vs. kg |

---

### 3.4 TS-EC-004: PM Calculation Errors

#### Symptoms
- Calculated PM emissions do not match stack test results
- PM values from opacity correlation are incorrect
- Filterable vs. condensable PM discrepancies

#### Diagnostic Steps

**Step 1: Check PM Calculation Method**

```bash
# Show PM calculation methodology
greenlang emissions methodology \
  --facility facility-001 \
  --pollutant PM \
  --show-basis

# Check opacity correlation
greenlang emissions opacity-correlation \
  --facility facility-001 \
  --unit boiler-001 \
  --show-coefficients
```

**Step 2: Verify Stack Test Data**

```bash
# Get stack test history
greenlang stacktest history \
  --facility facility-001 \
  --pollutant PM \
  --last 3y

# Check current PM emission factor
greenlang emissions emission-factors \
  --facility facility-001 \
  --pollutant PM \
  --show-source
```

**Step 3: Compare PM Components**

```bash
# Breakdown PM calculation
greenlang emissions pm-breakdown \
  --facility facility-001 \
  --unit boiler-001 \
  --time-range "2025-11-26T00:00:00Z/2025-11-26T23:59:59Z" \
  --show-filterable \
  --show-condensable
```

**Step 4: Validate Against Process Data**

```bash
# Compare PM to process indicators
greenlang emissions pm-correlation \
  --facility facility-001 \
  --unit boiler-001 \
  --correlate-with "load,ash_content,esp_power"
```

#### Common Causes and Solutions

| Cause | Symptoms | Solution |
|-------|----------|----------|
| Outdated stack test factor | Trending drift | Conduct new stack test |
| Wrong opacity correlation | Variable error | Recalibrate correlation |
| Missing condensable PM | Underestimate | Include CPM in calculation |
| Control device efficiency change | Step change | Update efficiency factor |
| Grain loading basis error | Factor error | Verify gr/dscf basis |
| Ash content change | Variable error | Update fuel ash data |

---

### 3.5 TS-EC-005: F-Factor Selection Issues

#### Symptoms
- F-factor used does not match fuel type
- Heat input calculation errors
- Emissions rate (lb/MMBtu) consistently off

#### Diagnostic Steps

**Step 1: Verify F-Factor Configuration**

```bash
# Show configured F-factors
greenlang config show \
  --facility facility-001 \
  --section "emissions.f_factors"

# Compare with EPA reference values
greenlang emissions f-factors \
  --facility facility-001 \
  --compare-with-epa
```

**Step 2: Check Fuel Type Mapping**

```bash
# Show fuel type to F-factor mapping
greenlang config show \
  --facility facility-001 \
  --section "fuels.f_factor_mapping"

# Verify fuel types in use
greenlang fuel types \
  --facility facility-001 \
  --active-only
```

**Step 3: Validate F-Factor Selection Logic**

```bash
# Trace F-factor selection
greenlang emissions trace-f-factor \
  --facility facility-001 \
  --unit boiler-001 \
  --time-range "2025-11-26T00:00:00Z/2025-11-26T23:59:59Z" \
  --show-selection-logic

# Check for fuel blending impact
greenlang emissions f-factor-blend \
  --facility facility-001 \
  --unit boiler-001 \
  --time-range "2025-11-26T00:00:00Z/2025-11-26T23:59:59Z"
```

#### Common Causes and Solutions

| Cause | Symptoms | Solution |
|-------|----------|----------|
| Wrong fuel type F-factor | Consistent offset | Correct fuel type mapping |
| Fc vs. Fd vs. Fw confusion | O2/CO2 related error | Use correct F-factor type |
| Custom fuel needs unique F-factor | Variable error | Calculate fuel-specific F-factor |
| Fuel blend not weighted | Error with multi-fuel | Implement weighted F-factor |

**Resolution Example:**

```bash
# Update F-factor for specific fuel
greenlang config update \
  --facility facility-001 \
  --path "emissions.f_factors.subbituminous_coal.Fd" \
  --value 9780 \
  --reason "Site-specific F-factor from fuel analysis"
```

---

### 3.6 TS-EC-006: Temperature/Pressure Correction Errors

#### Symptoms
- Corrected emissions differ significantly from uncorrected
- Stack flow calculations incorrect
- Standard vs. actual conditions mismatch

#### Diagnostic Steps

**Step 1: Verify T/P Sensor Data**

```bash
# Check temperature sensor readings
greenlang cems sensor-data \
  --facility facility-001 \
  --sensor stack_temperature \
  --time-range "last-24h" \
  --statistics

# Check pressure sensor readings
greenlang cems sensor-data \
  --facility facility-001 \
  --sensor stack_pressure \
  --time-range "last-24h" \
  --statistics
```

**Step 2: Check Standard Conditions Configuration**

```bash
# Show standard conditions configuration
greenlang config show \
  --facility facility-001 \
  --section "emissions.standard_conditions"

# Verify regulatory basis
greenlang compliance standard-conditions \
  --facility facility-001 \
  --show-regulatory-basis
```

**Step 3: Validate Correction Calculation**

```bash
# Trace T/P correction
greenlang emissions trace-tp-correction \
  --facility facility-001 \
  --unit boiler-001 \
  --timestamp "2025-11-26T10:00:00Z" \
  --show-formula

# Compare corrected vs. uncorrected
greenlang emissions compare-corrected \
  --facility facility-001 \
  --pollutant NOx \
  --time-range "2025-11-26T00:00:00Z/2025-11-26T23:59:59Z"
```

#### Common Causes and Solutions

| Cause | Symptoms | Solution |
|-------|----------|----------|
| Wrong standard temperature | Consistent offset | Verify 68F vs. 77F vs. 60F basis |
| Wrong standard pressure | Small consistent offset | Verify 29.92 vs. 30.0 inHg |
| Sensor failure | Erratic or flat readings | Replace/calibrate sensor |
| Units mismatch | Factor error | Verify K vs. C vs. F, atm vs. inHg |
| Altitude correction missing | Location-specific error | Apply altitude correction |

---

### 3.7 TS-EC-007: Unit Conversion Errors

#### Symptoms
- Values off by orders of magnitude
- Inconsistent units between systems
- Regulatory limits appear wrong

#### Diagnostic Steps

**Step 1: Identify Unit Configuration**

```bash
# Show all unit configurations
greenlang config show \
  --facility facility-001 \
  --section "units" \
  --include-all-pollutants

# Check CEMS native units
greenlang cems unit-configuration \
  --facility facility-001 \
  --all-parameters
```

**Step 2: Trace Unit Conversions**

```bash
# Trace conversion chain
greenlang emissions trace-conversion \
  --facility facility-001 \
  --pollutant NOx \
  --from-unit "ppm" \
  --to-unit "lb/hr" \
  --show-chain

# Verify conversion factors
greenlang emissions conversion-factors \
  --pollutant NOx \
  --all-conversions
```

**Step 3: Test Conversion Calculation**

```bash
# Test specific conversion
greenlang emissions test-conversion \
  --value 50 \
  --from-unit "ppm" \
  --to-unit "lb/hr" \
  --flow-rate 100000 \
  --flow-unit "scfm" \
  --molecular-weight 46
```

#### Common Causes and Solutions

| Cause | Symptoms | Solution |
|-------|----------|----------|
| ppm to mass confusion | Large factor error | Use correct molecular weight |
| Standard vs. actual flow | ~10-20% error | Verify flow correction |
| Short ton vs. metric ton | 9% error | Clarify ton type |
| scfm vs. acfm | Variable error | Correct flow basis |
| lb/MMBtu to ppm mismatch | Complex error | Verify conversion includes F-factor |

---

### 3.8 TS-EC-008: Averaging Period Calculation Issues

#### Symptoms
- Rolling averages do not match manual calculations
- Averaging period violations triggered incorrectly
- Hourly vs. block average discrepancies

#### Diagnostic Steps

**Step 1: Verify Averaging Period Configuration**

```bash
# Show averaging period configuration
greenlang config show \
  --facility facility-001 \
  --section "emissions.averaging_periods"

# Check permit requirements
greenlang compliance averaging-requirements \
  --facility facility-001 \
  --pollutant NOx
```

**Step 2: Trace Averaging Calculation**

```bash
# Show averaging calculation details
greenlang emissions trace-average \
  --facility facility-001 \
  --pollutant NOx \
  --averaging-period "1-hour" \
  --timestamp "2025-11-26T10:00:00Z" \
  --show-all-inputs

# Check rolling average calculation
greenlang emissions rolling-average \
  --facility facility-001 \
  --pollutant NOx \
  --period "30-day" \
  --end-date "2025-11-26" \
  --show-calculation
```

**Step 3: Validate Input Data Points**

```bash
# Check data completeness for averaging period
greenlang cems data-completeness \
  --facility facility-001 \
  --pollutant NOx \
  --period "2025-11-26T09:00:00Z/2025-11-26T10:00:00Z"

# Check for substitute data in period
greenlang cems substitute-data-usage \
  --facility facility-001 \
  --pollutant NOx \
  --period "2025-11-26T00:00:00Z/2025-11-26T23:59:59Z"
```

#### Common Causes and Solutions

| Cause | Symptoms | Solution |
|-------|----------|----------|
| Clock hour vs. rolling confusion | Timing differences | Align with permit definition |
| Block average vs. hourly | Systematic difference | Use correct averaging method |
| Incomplete data handling | High/low averages | Apply correct data substitution |
| Timezone issues | Shifted averages | Verify timezone configuration |
| Arithmetic vs. weighted mean | For variable flow | Use flow-weighted if required |

---

## 4. CEMS Integration Issues

### 4.1 TS-CEMS-001: Modbus Connection Timeout

#### Symptoms
- No data received from CEMS
- Connection timeout errors in logs
- Intermittent data gaps

#### Diagnostic Steps

**Step 1: Basic Connectivity Check**

```bash
# Test Modbus connectivity
greenlang cems test-connection \
  --facility facility-001 \
  --protocol modbus \
  --verbose

# Check network path
greenlang network trace \
  --from gl-010-pod \
  --to cems-server:502 \
  --protocol tcp
```

**Step 2: Protocol-Level Diagnostics**

```bash
# Test Modbus read operation
greenlang cems modbus-test \
  --facility facility-001 \
  --address 192.168.1.100 \
  --port 502 \
  --unit-id 1 \
  --register 40001 \
  --count 10

# Check Modbus configuration
greenlang config show \
  --facility facility-001 \
  --section "cems.modbus"
```

**Step 3: Check CEMS Device Status**

```bash
# Get CEMS device diagnostic information
greenlang cems device-info \
  --facility facility-001 \
  --address 192.168.1.100

# Check for Modbus exception responses
greenlang cems modbus-errors \
  --facility facility-001 \
  --time-range "last-24h"
```

#### Common Causes and Solutions

| Cause | Symptoms | Solution |
|-------|----------|----------|
| Network firewall blocking | No connection | Open port 502 |
| CEMS device busy | Intermittent timeout | Increase polling interval |
| Wrong Modbus address | Connection refused | Verify device address |
| TCP keep-alive timeout | Dropped connections | Adjust keep-alive settings |
| Modbus RTU vs. TCP confusion | Protocol errors | Match protocol configuration |
| Unit ID mismatch | No response | Correct unit ID setting |

**Resolution Example:**

```bash
# Update Modbus configuration
greenlang config update \
  --facility facility-001 \
  --path "cems.modbus.timeout_ms" \
  --value 5000 \
  --reason "Increase timeout for slow CEMS response"

# Restart connector
greenlang connector restart modbus-cems-facility-001
```

---

### 4.2 TS-CEMS-002: OPC-UA Authentication Failure

#### Symptoms
- Authentication rejected errors
- Certificate validation failures
- Session establishment failures

#### Diagnostic Steps

**Step 1: Test OPC-UA Connection**

```bash
# Test OPC-UA connectivity
greenlang cems test-connection \
  --facility facility-001 \
  --protocol opc-ua \
  --verbose

# Check certificate status
greenlang cems opc-ua-certificates \
  --facility facility-001 \
  --check-validity
```

**Step 2: Verify Authentication Configuration**

```bash
# Show OPC-UA configuration
greenlang config show \
  --facility facility-001 \
  --section "cems.opc_ua"

# Test authentication
greenlang cems opc-ua-auth-test \
  --facility facility-001 \
  --endpoint "opc.tcp://cems-server:4840" \
  --security-mode SignAndEncrypt
```

**Step 3: Check Server Requirements**

```bash
# Browse OPC-UA server security policies
greenlang cems opc-ua-browse-security \
  --endpoint "opc.tcp://cems-server:4840"

# Check server certificate
greenlang cems opc-ua-server-cert \
  --endpoint "opc.tcp://cems-server:4840" \
  --output /tmp/server-cert.pem
```

#### Common Causes and Solutions

| Cause | Symptoms | Solution |
|-------|----------|----------|
| Expired certificate | Certificate validation failed | Renew certificate |
| Wrong security policy | Connection rejected | Match server security policy |
| Certificate not trusted | Trust chain error | Add server cert to trust store |
| Username/password wrong | Authentication failed | Update credentials |
| IP address in certificate | Certificate hostname mismatch | Use correct endpoint URL |

**Resolution Example:**

```bash
# Renew OPC-UA certificate
greenlang cems opc-ua-renew-cert \
  --facility facility-001 \
  --validity-days 365

# Update trust store
greenlang cems opc-ua-trust \
  --facility facility-001 \
  --server-cert /tmp/server-cert.pem
```

---

### 4.3 TS-CEMS-003: Data Quality Flags Interpretation

#### Symptoms
- Wrong data marked as valid/invalid
- Quality flags not being applied correctly
- Substitute data triggered incorrectly

#### Diagnostic Steps

**Step 1: Review Quality Flag Configuration**

```bash
# Show quality flag mappings
greenlang config show \
  --facility facility-001 \
  --section "cems.quality_flags"

# Check CEMS-specific flags
greenlang cems quality-flag-values \
  --facility facility-001 \
  --list-all
```

**Step 2: Analyze Quality Flag Data**

```bash
# Get quality flag statistics
greenlang cems quality-flag-stats \
  --facility facility-001 \
  --time-range "last-7d"

# Trace quality flag processing
greenlang cems trace-quality-flags \
  --facility facility-001 \
  --timestamp "2025-11-26T10:00:00Z" \
  --show-source-flags \
  --show-processed-flags
```

**Step 3: Compare Source and Processed**

```bash
# Compare CEMS quality flags with GL-010 interpretation
greenlang cems compare-quality-flags \
  --facility facility-001 \
  --time-range "2025-11-26T00:00:00Z/2025-11-26T23:59:59Z" \
  --show-mismatches
```

#### Common Causes and Solutions

| Cause | Symptoms | Solution |
|-------|----------|----------|
| Flag mapping incorrect | Systematic misinterpretation | Update flag mapping |
| Vendor-specific flags | Unknown flags ignored | Add vendor flag definitions |
| Flag priority wrong | Wrong quality applied | Adjust flag priority |
| Composite flag handling | Complex scenarios wrong | Review composite logic |

---

### 4.4 TS-CEMS-004: Missing Data Substitution Errors

#### Symptoms
- Wrong substitute values calculated
- Substitute data not applied when required
- Over-substitution affecting compliance

#### Diagnostic Steps

**Step 1: Review Substitution Configuration**

```bash
# Show substitution rules
greenlang config show \
  --facility facility-001 \
  --section "cems.data_substitution"

# Check regulatory program requirements
greenlang compliance substitution-requirements \
  --facility facility-001 \
  --program "Part75"
```

**Step 2: Trace Substitution Logic**

```bash
# Trace specific substitution event
greenlang cems trace-substitution \
  --facility facility-001 \
  --pollutant NOx \
  --timestamp "2025-11-26T10:00:00Z" \
  --show-inputs \
  --show-lookback

# Check substitution statistics
greenlang cems substitution-stats \
  --facility facility-001 \
  --period "Q4-2025"
```

**Step 3: Verify Lookback Data**

```bash
# Check lookback data availability
greenlang cems lookback-data \
  --facility facility-001 \
  --pollutant NOx \
  --lookback-hours 2160 \
  --statistics

# Verify percentile calculation
greenlang cems percentile-data \
  --facility facility-001 \
  --pollutant NOx \
  --percentile 90 \
  --period "last-2160h"
```

#### Common Causes and Solutions

| Cause | Symptoms | Solution |
|-------|----------|----------|
| Wrong lookback period | Incorrect percentile | Configure correct lookback |
| Missing historical data | Substitution fails | Backfill historical data |
| Wrong percentile | High/low substitute values | Use regulation-specific percentile |
| Load bin not configured | Substitute ignores load | Configure load-based substitution |
| Maximum potential not set | Extreme substitute values | Configure maximum potential |

---

### 4.5 TS-CEMS-005: Calibration Detection False Positives

#### Symptoms
- Normal operations flagged as calibration
- Calibration gas injection not detected
- Wrong data excluded from compliance

#### Diagnostic Steps

**Step 1: Review Calibration Detection Rules**

```bash
# Show calibration detection configuration
greenlang config show \
  --facility facility-001 \
  --section "cems.calibration_detection"

# Check detection thresholds
greenlang cems calibration-thresholds \
  --facility facility-001 \
  --pollutant NOx
```

**Step 2: Analyze Detection Events**

```bash
# Review recent calibration detections
greenlang cems calibration-events \
  --facility facility-001 \
  --time-range "last-7d" \
  --show-detection-reason

# Compare with actual calibration schedule
greenlang cems calibration-schedule \
  --facility facility-001 \
  --time-range "last-7d"
```

**Step 3: Analyze Signal Patterns**

```bash
# Analyze signal during detected calibrations
greenlang cems signal-analysis \
  --facility facility-001 \
  --pollutant NOx \
  --event-type calibration \
  --time-range "2025-11-26T06:00:00Z/2025-11-26T07:00:00Z"
```

#### Common Causes and Solutions

| Cause | Symptoms | Solution |
|-------|----------|----------|
| Threshold too sensitive | False positives | Adjust detection thresholds |
| Process upset looks like cal | Random false positives | Add process context checks |
| Cal flag from CEMS missed | Cal not detected | Check flag acquisition |
| Duration threshold wrong | Short cals missed | Adjust duration threshold |

---

### 4.6 TS-CEMS-006: Multi-Analyzer Synchronization

#### Symptoms
- Primary and backup analyzer values don't match
- Switching between analyzers causes data discontinuity
- Comparison reports show drift between analyzers

#### Diagnostic Steps

**Step 1: Compare Analyzer Outputs**

```bash
# Compare primary and backup analyzer
greenlang cems compare-analyzers \
  --facility facility-001 \
  --pollutant NOx \
  --time-range "last-24h" \
  --statistics

# Check analyzer synchronization status
greenlang cems sync-status \
  --facility facility-001 \
  --pollutant NOx
```

**Step 2: Check Individual Analyzer Calibrations**

```bash
# Primary analyzer calibration status
greenlang cems calibration-status \
  --analyzer-id NOx-analyzer-001 \
  --detail-level full

# Backup analyzer calibration status
greenlang cems calibration-status \
  --analyzer-id NOx-analyzer-002 \
  --detail-level full
```

**Step 3: Analyze Switching Events**

```bash
# Review analyzer switching history
greenlang cems switch-history \
  --facility facility-001 \
  --pollutant NOx \
  --time-range "last-30d"

# Analyze data during switches
greenlang cems switch-impact \
  --facility facility-001 \
  --pollutant NOx \
  --show-discontinuities
```

#### Common Causes and Solutions

| Cause | Symptoms | Solution |
|-------|----------|----------|
| Calibration drift | Gradual divergence | Calibrate both analyzers |
| Sample system difference | Consistent offset | Verify common sample point |
| Response time difference | Dynamic difference | Adjust sampling timing |
| Reference gas difference | Offset proportional to span | Use same reference gas |

---

### 4.7 TS-CEMS-007: Stack Flow Calculation Errors

#### Symptoms
- Mass emissions calculations incorrect
- Flow rate values don't match ultrasonic/differential pressure calculation
- Volumetric vs. mass flow confusion

#### Diagnostic Steps

**Step 1: Verify Flow Calculation Method**

```bash
# Show flow calculation configuration
greenlang config show \
  --facility facility-001 \
  --section "cems.flow_calculation"

# Check flow measurement type
greenlang cems flow-measurement-type \
  --facility facility-001 \
  --unit boiler-001
```

**Step 2: Check Flow Input Parameters**

```bash
# Get flow sensor readings
greenlang cems flow-parameters \
  --facility facility-001 \
  --unit boiler-001 \
  --time-range "last-24h" \
  --parameters "velocity,temperature,pressure,moisture"

# Check stack geometry
greenlang config show \
  --facility facility-001 \
  --section "cems.stack_geometry"
```

**Step 3: Trace Flow Calculation**

```bash
# Trace flow calculation
greenlang cems trace-flow \
  --facility facility-001 \
  --unit boiler-001 \
  --timestamp "2025-11-26T10:00:00Z" \
  --show-formula \
  --show-inputs

# Compare with load-based expected flow
greenlang cems flow-vs-load \
  --facility facility-001 \
  --unit boiler-001 \
  --time-range "last-24h"
```

#### Common Causes and Solutions

| Cause | Symptoms | Solution |
|-------|----------|----------|
| Stack area wrong | Proportional error | Verify stack dimensions |
| Velocity profile incorrect | Consistent offset | Check K-factor from flow RATA |
| Moisture correction | Error varies with conditions | Verify moisture measurement |
| Standard conditions | Different basis | Align standard conditions |

---

## 5. Compliance Checking Issues

### 5.1 TS-CC-001: Wrong Regulatory Limit Applied

#### Symptoms
- Compliance status incorrect
- Wrong limit shown in reports
- Exceedance alerts triggered incorrectly

#### Diagnostic Steps

**Step 1: Verify Limit Configuration**

```bash
# Show configured limits
greenlang compliance limits \
  --facility facility-001 \
  --pollutant NOx \
  --all-limits

# Check permit database
greenlang compliance permit-limits \
  --facility facility-001 \
  --permit-number "AIR-2024-001" \
  --pollutant NOx
```

**Step 2: Check Limit Selection Logic**

```bash
# Trace limit selection
greenlang compliance trace-limit-selection \
  --facility facility-001 \
  --unit boiler-001 \
  --pollutant NOx \
  --timestamp "2025-11-26T10:00:00Z" \
  --show-candidates \
  --show-selection-reason

# Check operating mode impact
greenlang compliance limit-by-mode \
  --facility facility-001 \
  --unit boiler-001 \
  --modes "normal,startup,shutdown"
```

**Step 3: Verify Effective Dates**

```bash
# Check limit effective dates
greenlang compliance limit-history \
  --facility facility-001 \
  --pollutant NOx \
  --show-effective-dates

# Check for pending limit changes
greenlang compliance pending-limit-changes \
  --facility facility-001
```

#### Common Causes and Solutions

| Cause | Symptoms | Solution |
|-------|----------|----------|
| Outdated permit data | Wrong limit | Update permit database |
| Wrong unit assigned | Wrong limit for unit | Correct unit assignment |
| Operating mode not detected | Wrong mode limit | Fix mode detection |
| Effective date wrong | Limit timing incorrect | Correct effective dates |
| Multi-jurisdiction conflict | Inconsistent limits | Apply most stringent |

---

### 5.2 TS-CC-002: Averaging Period Mismatch

#### Symptoms
- Compliance calculated for wrong period
- Regulatory report shows wrong averaging basis
- Exceedance detection timing incorrect

#### Diagnostic Steps

**Step 1: Verify Averaging Period Configuration**

```bash
# Show averaging period configuration
greenlang compliance averaging-periods \
  --facility facility-001 \
  --pollutant NOx \
  --show-all

# Check permit requirements
greenlang compliance permit-averaging \
  --facility facility-001 \
  --permit-number "AIR-2024-001"
```

**Step 2: Trace Averaging Period Application**

```bash
# Trace averaging period selection
greenlang compliance trace-averaging \
  --facility facility-001 \
  --pollutant NOx \
  --timestamp "2025-11-26T10:00:00Z" \
  --show-period-calculation

# Check multiple averaging periods
greenlang compliance multi-period-status \
  --facility facility-001 \
  --pollutant NOx \
  --periods "1h,3h,24h,30d"
```

#### Common Causes and Solutions

| Cause | Symptoms | Solution |
|-------|----------|----------|
| Block vs. rolling confusion | Timing differences | Match permit definition |
| Clock hour alignment | Off-by-minutes | Verify timezone and alignment |
| Calendar vs. operating | Count differences | Apply correct period type |

---

### 5.3 TS-CC-003: Startup/Shutdown Exemption Handling

#### Symptoms
- Exemptions not applied during startup/shutdown
- Exemptions applied when not applicable
- Duration limits exceeded

#### Diagnostic Steps

**Step 1: Verify Exemption Configuration**

```bash
# Show startup/shutdown exemption rules
greenlang compliance exemption-rules \
  --facility facility-001 \
  --type "startup-shutdown"

# Check permit language
greenlang compliance permit-exemptions \
  --facility facility-001 \
  --permit-number "AIR-2024-001"
```

**Step 2: Trace Exemption Application**

```bash
# Trace exemption for specific event
greenlang compliance trace-exemption \
  --facility facility-001 \
  --unit boiler-001 \
  --event-type startup \
  --timestamp "2025-11-26T06:00:00Z" \
  --show-criteria-check

# Check startup/shutdown detection
greenlang process detect-mode \
  --facility facility-001 \
  --unit boiler-001 \
  --time-range "2025-11-26T05:00:00Z/2025-11-26T08:00:00Z"
```

**Step 3: Verify Duration Tracking**

```bash
# Check exemption duration usage
greenlang compliance exemption-duration \
  --facility facility-001 \
  --unit boiler-001 \
  --period "annual" \
  --year 2025
```

#### Common Causes and Solutions

| Cause | Symptoms | Solution |
|-------|----------|----------|
| Mode detection wrong | Exemption timing wrong | Adjust mode detection parameters |
| Duration tracking error | Duration exceeded | Fix duration counter |
| Notification not sent | Exemption invalidated | Configure notifications |
| Wrong permit conditions | Exemption criteria wrong | Update permit configuration |

---

### 5.4 TS-CC-004: Malfunction Provision Interpretation

#### Symptoms
- Malfunction exemption not granted when applicable
- Exemption granted for non-qualifying events
- Documentation requirements not met

#### Diagnostic Steps

**Step 1: Review Malfunction Criteria**

```bash
# Show malfunction exemption criteria
greenlang compliance malfunction-criteria \
  --facility facility-001 \
  --regulatory-program "NESHAP"

# Check documented malfunctions
greenlang compliance malfunction-log \
  --facility facility-001 \
  --time-range "last-90d"
```

**Step 2: Verify Malfunction Qualification**

```bash
# Trace malfunction determination
greenlang compliance trace-malfunction \
  --facility facility-001 \
  --incident-id "MAL-2025-001" \
  --show-criteria-evaluation

# Check response timing
greenlang compliance malfunction-response \
  --facility facility-001 \
  --incident-id "MAL-2025-001" \
  --check-timeline
```

#### Common Causes and Solutions

| Cause | Symptoms | Solution |
|-------|----------|----------|
| Response timing wrong | Exemption denied | Document response timeline |
| Predictable event | Doesn't qualify | Review maintenance practices |
| Documentation incomplete | Cannot verify | Complete required documentation |
| Repair not minimized | Exemption limited | Document repair efforts |

---

### 5.5 TS-CC-005: Multi-Jurisdiction Conflict

#### Symptoms
- Different agencies showing different compliance status
- Conflicting limits applied
- Report requirements conflict

#### Diagnostic Steps

**Step 1: Identify All Applicable Jurisdictions**

```bash
# Show all applicable regulations
greenlang compliance jurisdictions \
  --facility facility-001 \
  --show-all

# Check for conflicts
greenlang compliance jurisdiction-conflicts \
  --facility facility-001 \
  --pollutant NOx
```

**Step 2: Compare Requirements**

```bash
# Compare limits across jurisdictions
greenlang compliance compare-limits \
  --facility facility-001 \
  --pollutant NOx \
  --jurisdictions "federal,state,local"

# Compare reporting requirements
greenlang compliance compare-reporting \
  --facility facility-001 \
  --jurisdictions "federal,state,local"
```

**Step 3: Determine Applicable Limits**

```bash
# Apply most stringent rule
greenlang compliance most-stringent \
  --facility facility-001 \
  --pollutant NOx \
  --show-rationale
```

#### Common Causes and Solutions

| Cause | Symptoms | Solution |
|-------|----------|----------|
| New regulation not added | Missing jurisdiction | Add regulation to configuration |
| SIP update not reflected | Outdated state limits | Update state requirements |
| Local rule override | Stricter limit missed | Add local requirements |

---

## 6. Reporting Issues

### 6.1 TS-RPT-001: XML Schema Validation Failures

#### Symptoms
- Report submission rejected
- Schema validation errors
- XML structure invalid

#### Diagnostic Steps

**Step 1: Identify Validation Errors**

```bash
# Run schema validation
greenlang report validate \
  --report-id quarterly-2025-Q3-facility-001 \
  --schema "EPA-ERT-v3.0" \
  --output-errors detailed

# Get error locations
greenlang report validation-errors \
  --report-id quarterly-2025-Q3-facility-001 \
  --show-line-numbers
```

**Step 2: Check Schema Version**

```bash
# Verify schema version
greenlang report schema-info \
  --report-id quarterly-2025-Q3-facility-001

# Check for schema updates
greenlang report schema-updates \
  --program "EPA-CEDRI" \
  --since "2025-01-01"
```

**Step 3: Analyze XML Structure**

```bash
# Export report XML for analysis
greenlang report export-xml \
  --report-id quarterly-2025-Q3-facility-001 \
  --output /tmp/report.xml

# Validate against specific schema
greenlang report validate-xml \
  --input /tmp/report.xml \
  --schema /schemas/EPA-ERT-v3.0.xsd
```

#### Common Causes and Solutions

| Cause | Symptoms | Solution |
|-------|----------|----------|
| Schema version mismatch | Namespace errors | Use correct schema version |
| Required element missing | Missing element error | Add required data |
| Invalid data type | Type validation error | Fix data format |
| Invalid enumeration | Value not allowed | Use allowed value |
| Namespace error | Namespace mismatch | Fix namespace declaration |

**Resolution Example:**

```bash
# Auto-fix common XML issues
greenlang report auto-fix \
  --report-id quarterly-2025-Q3-facility-001 \
  --fix-types "format,namespace,enumeration"

# Regenerate with correct schema
greenlang report regenerate \
  --report-id quarterly-2025-Q3-facility-001 \
  --schema-version "3.0"
```

---

### 6.2 TS-RPT-002: EPA CDX Submission Errors

#### Symptoms
- CDX submission fails
- Authentication errors
- Submission status unknown

#### Diagnostic Steps

**Step 1: Check CDX Connectivity**

```bash
# Test CDX connection
greenlang integration test EPA_CDX --verbose

# Check CDX status page
greenlang web-check https://cdx.epa.gov/status
```

**Step 2: Verify Credentials**

```bash
# Test CDX authentication
greenlang integration auth-test EPA_CDX

# Check credential expiration
greenlang integration credential-status EPA_CDX
```

**Step 3: Check Submission Status**

```bash
# Get submission history
greenlang report submission-history \
  --portal EPA_CDX \
  --facility facility-001 \
  --last 30d

# Check specific submission
greenlang report submission-status \
  --submission-id "CDX-2025-12345"
```

#### Common Causes and Solutions

| Cause | Symptoms | Solution |
|-------|----------|----------|
| CDX maintenance | Connection refused | Wait for maintenance end |
| Credential expired | Authentication failed | Renew CDX credentials |
| NAAS issue | SSO failure | Clear session and retry |
| Large file timeout | Submission timeout | Split or compress file |
| Network issue | Connection timeout | Check network connectivity |

---

### 6.3 TS-RPT-003: EU ETS Verification Failures

#### Symptoms
- Verification statement rejected
- Data discrepancy with verifier
- Registry submission fails

#### Diagnostic Steps

**Step 1: Check Verification Requirements**

```bash
# Show verification status
greenlang ets verification-status \
  --installation-id EU-DE-123456 \
  --year 2024

# Check verifier accreditation
greenlang ets verifier-check \
  --installation-id EU-DE-123456
```

**Step 2: Compare Data with Verifier**

```bash
# Export verification data package
greenlang ets export-verification-data \
  --installation-id EU-DE-123456 \
  --year 2024 \
  --output /exports/verification-2024/

# Check data reconciliation
greenlang ets reconcile \
  --installation-id EU-DE-123456 \
  --verifier-data /imports/verifier-data-2024.xlsx
```

**Step 3: Check Registry Submission**

```bash
# Test registry connection
greenlang integration test EU_ETS_REGISTRY --verbose

# Check submission format
greenlang ets validate-submission \
  --installation-id EU-DE-123456 \
  --year 2024
```

#### Common Causes and Solutions

| Cause | Symptoms | Solution |
|-------|----------|----------|
| Data mismatch with verifier | Verification fails | Reconcile data differences |
| Signature issue | Statement rejected | Request new statement |
| Registry maintenance | Submission fails | Retry after maintenance |
| Methodology disagreement | Non-conformity | Resolve with verifier |

---

### 6.4 TS-RPT-004: Digital Signature Issues

#### Symptoms
- Report cannot be signed
- Signature validation fails
- Certificate errors

#### Diagnostic Steps

**Step 1: Check Certificate Status**

```bash
# Check signing certificate
greenlang certificate status \
  --purpose report-signing \
  --show-expiration

# Validate certificate chain
greenlang certificate validate-chain \
  --purpose report-signing
```

**Step 2: Test Signature Operation**

```bash
# Test signing capability
greenlang signature test \
  --certificate-id reporting-cert-001 \
  --test-data "test message"

# Check signature configuration
greenlang config show \
  --section "reporting.digital_signature"
```

**Step 3: Verify Signature on Report**

```bash
# Verify existing signature
greenlang signature verify \
  --report-id quarterly-2025-Q3-facility-001

# Check signature timestamps
greenlang signature timestamp-info \
  --report-id quarterly-2025-Q3-facility-001
```

#### Common Causes and Solutions

| Cause | Symptoms | Solution |
|-------|----------|----------|
| Certificate expired | Signing fails | Renew certificate |
| Wrong certificate type | Signature rejected | Use correct certificate |
| Timestamping service down | Timestamp missing | Use alternate TSA |
| Key access denied | Permission error | Check key permissions |

---

### 6.5 TS-RPT-005: Report Formatting Errors

#### Symptoms
- Report appears incorrectly formatted
- PDF generation fails
- Data appears in wrong location

#### Diagnostic Steps

**Step 1: Check Report Template**

```bash
# Verify template configuration
greenlang report template-info \
  --report-type quarterly-emissions

# Check template version
greenlang report template-version \
  --report-type quarterly-emissions
```

**Step 2: Test Report Generation**

```bash
# Generate test report
greenlang report generate \
  --type quarterly-emissions \
  --facility facility-001 \
  --period Q3-2025 \
  --output-format pdf \
  --test-mode

# Compare with expected format
greenlang report compare-format \
  --generated /tmp/test-report.pdf \
  --expected /templates/expected-format.pdf
```

**Step 3: Check Data Mapping**

```bash
# Verify data field mapping
greenlang report field-mapping \
  --report-type quarterly-emissions \
  --show-all

# Check for unmapped fields
greenlang report unmapped-fields \
  --report-id quarterly-2025-Q3-facility-001
```

#### Common Causes and Solutions

| Cause | Symptoms | Solution |
|-------|----------|----------|
| Template outdated | Wrong format | Update template |
| Field mapping wrong | Data in wrong place | Fix field mapping |
| Font missing | PDF rendering issue | Install required fonts |
| Data overflow | Truncated data | Adjust field width |

---

## 7. Diagnostic Commands Reference

### 7.1 System Health Commands

```bash
# Full system health check
greenlang health --agent GL-010 --full

# Component-specific health
greenlang health --agent GL-010 --component emissions-calculator
greenlang health --agent GL-010 --component cems-integration
greenlang health --agent GL-010 --component compliance-engine
greenlang health --agent GL-010 --component report-generator

# Dependency health
greenlang dependencies status --agent GL-010

# Database health
greenlang db health --database gl010-emissions-db
```

### 7.2 CEMS Diagnostic Commands

```bash
# CEMS system status
greenlang cems status --facility {facility-id}

# Analyzer diagnostics
greenlang cems analyzer-diagnostic --analyzer-id {analyzer-id} --full

# Data quality analysis
greenlang cems data-quality --facility {facility-id} --period {period}

# Communication diagnostics
greenlang cems communication-test --facility {facility-id} --verbose

# Calibration status
greenlang cems calibration-status --facility {facility-id}

# Data flow verification
greenlang cems verify-data-flow --facility {facility-id} --duration 5m
```

### 7.3 Emissions Calculation Commands

```bash
# Trace calculation
greenlang emissions trace --facility {facility-id} --pollutant {pollutant} --timestamp {timestamp}

# Calculation verification
greenlang emissions verify --facility {facility-id} --pollutant {pollutant} --time-range {range}

# Compare calculation methods
greenlang emissions compare-methods --facility {facility-id} --methods "cems,fuel-based"

# Emission factor check
greenlang emissions emission-factors --facility {facility-id} --all

# Unit conversion test
greenlang emissions test-conversion --value {value} --from-unit {from} --to-unit {to}
```

### 7.4 Compliance Diagnostic Commands

```bash
# Compliance status
greenlang compliance status --facility {facility-id}

# Limit verification
greenlang compliance verify-limits --facility {facility-id} --pollutant {pollutant}

# Exemption check
greenlang compliance exemption-check --facility {facility-id} --event-type {type}

# Regulatory requirement check
greenlang compliance requirements --facility {facility-id} --jurisdiction {jurisdiction}

# Averaging period status
greenlang compliance averaging-status --facility {facility-id} --all-periods
```

### 7.5 Reporting Diagnostic Commands

```bash
# Report validation
greenlang report validate --report-id {report-id} --schema {schema}

# Submission status
greenlang report submission-status --report-id {report-id}

# Data completeness check
greenlang report data-completeness --report-id {report-id}

# Integration status
greenlang integration status {integration-name}

# Certificate status
greenlang certificate status --purpose report-signing
```

### 7.6 Infrastructure Diagnostic Commands

```bash
# Pod status
kubectl get pods -n gl-agents -l app=gl-010-emissionwatch

# Pod logs
kubectl logs -n gl-agents -l app=gl-010-emissionwatch --tail=500

# Resource usage
kubectl top pod -n gl-agents -l app=gl-010-emissionwatch

# Network connectivity
greenlang network test --from gl-010-pod --to {target}

# Database connectivity
greenlang db test-connection --database gl010-emissions-db
```

---

## 8. Common Error Messages

### 8.1 Emissions Calculation Errors

| Error Code | Message | Cause | Resolution |
|------------|---------|-------|------------|
| EC-001 | "F-factor not found for fuel type" | Missing F-factor configuration | Add F-factor for fuel type |
| EC-002 | "Unit conversion failed: unknown unit" | Invalid unit specified | Check unit configuration |
| EC-003 | "Division by zero in calculation" | Zero divisor (flow, O2, etc.) | Check input data validity |
| EC-004 | "Emission factor expired" | Stack test factor too old | Conduct new stack test |
| EC-005 | "Missing required input: {parameter}" | Required data not available | Verify data source |
| EC-006 | "Calculation result out of range" | Result exceeds expected bounds | Check inputs and formula |
| EC-007 | "Averaging period incomplete" | Not enough data for average | Wait for complete period |
| EC-008 | "Conflicting emission factors" | Multiple factors for same source | Resolve factor conflict |

### 8.2 CEMS Integration Errors

| Error Code | Message | Cause | Resolution |
|------------|---------|-------|------------|
| CEMS-001 | "Connection timeout to CEMS" | Network or CEMS unavailable | Check connectivity |
| CEMS-002 | "Authentication failed" | Invalid credentials | Update credentials |
| CEMS-003 | "Invalid data quality flag" | Unrecognized flag value | Update flag mapping |
| CEMS-004 | "Calibration drift exceeded limit" | Analyzer needs calibration | Perform calibration |
| CEMS-005 | "Missing data exceeds threshold" | Too much missing data | Apply substitute data |
| CEMS-006 | "Invalid register address" | Modbus config wrong | Fix register mapping |
| CEMS-007 | "Sample rate too slow" | Not meeting data requirements | Increase sample rate |
| CEMS-008 | "Analyzer offline" | Analyzer not responding | Check analyzer status |

### 8.3 Compliance Checking Errors

| Error Code | Message | Cause | Resolution |
|------------|---------|-------|------------|
| CC-001 | "Limit not found for unit/pollutant" | Missing limit configuration | Configure limit |
| CC-002 | "Invalid averaging period" | Period not supported | Use valid period |
| CC-003 | "Exemption criteria not met" | Exemption denied | Review criteria |
| CC-004 | "Regulatory program conflict" | Multiple programs disagree | Apply most stringent |
| CC-005 | "Permit expired" | Operating under expired permit | Renew permit |
| CC-006 | "Invalid operating mode" | Mode not recognized | Update mode definitions |
| CC-007 | "Compliance calculation failed" | Error in compliance logic | Check calculation inputs |

### 8.4 Reporting Errors

| Error Code | Message | Cause | Resolution |
|------------|---------|-------|------------|
| RPT-001 | "Schema validation failed" | XML doesn't match schema | Fix XML structure |
| RPT-002 | "Required field missing: {field}" | Missing required data | Populate required field |
| RPT-003 | "Submission rejected by portal" | Portal rejected submission | Check rejection reason |
| RPT-004 | "Digital signature failed" | Signing error | Check certificate |
| RPT-005 | "Report generation timeout" | Report too complex/large | Optimize or split report |
| RPT-006 | "Template not found" | Missing report template | Install template |
| RPT-007 | "Data export failed" | Export error | Check export configuration |
| RPT-008 | "Deadline passed" | Report overdue | Submit late notification |

### 8.5 System Errors

| Error Code | Message | Cause | Resolution |
|------------|---------|-------|------------|
| SYS-001 | "Database connection failed" | DB unavailable | Check database status |
| SYS-002 | "Out of memory" | Memory exhausted | Increase memory limit |
| SYS-003 | "Cache corruption detected" | Cache inconsistent | Clear and rebuild cache |
| SYS-004 | "Configuration load failed" | Invalid configuration | Fix configuration file |
| SYS-005 | "Service dependency unavailable" | Required service down | Check dependent services |
| SYS-006 | "Rate limit exceeded" | Too many API calls | Implement backoff |
| SYS-007 | "Disk space critical" | Storage full | Clean up or expand storage |
| SYS-008 | "Health check failed" | Component unhealthy | Investigate component |

---

## 9. Log Analysis

### 9.1 Log Locations

| Log Type | Location | Retention |
|----------|----------|-----------|
| Application logs | /var/log/gl010/app.log | 30 days |
| CEMS integration logs | /var/log/gl010/cems.log | 90 days |
| Calculation logs | /var/log/gl010/calculations.log | 7 years |
| Compliance logs | /var/log/gl010/compliance.log | 7 years |
| Audit logs | /var/log/gl010/audit.log | 7 years |
| Error logs | /var/log/gl010/error.log | 90 days |

### 9.2 Log Query Commands

```bash
# Search logs for errors
greenlang logs search \
  --agent GL-010 \
  --level error \
  --time-range "last-1h"

# Search by facility
greenlang logs search \
  --agent GL-010 \
  --facility facility-001 \
  --time-range "last-24h"

# Search by incident
greenlang logs search \
  --agent GL-010 \
  --correlation-id {incident-id} \
  --all-levels

# Search by error code
greenlang logs search \
  --agent GL-010 \
  --pattern "EC-001" \
  --time-range "last-7d"

# Export logs for analysis
greenlang logs export \
  --agent GL-010 \
  --time-range "2025-11-26T00:00:00Z/2025-11-26T23:59:59Z" \
  --output /exports/logs-20251126.json
```

### 9.3 Log Patterns to Watch

**Critical Patterns:**
```
Pattern: "emissions.*exceeded.*limit"
Meaning: Emissions violation detected
Action: Investigate immediately

Pattern: "CEMS.*offline|unavailable"
Meaning: CEMS data loss
Action: Check CEMS connectivity

Pattern: "compliance.*violation"
Meaning: Compliance issue detected
Action: Review compliance status

Pattern: "report.*failed|rejected"
Meaning: Regulatory report issue
Action: Check report status
```

**Warning Patterns:**
```
Pattern: "calibration.*drift"
Meaning: Analyzer calibration drift
Action: Schedule calibration

Pattern: "data.*quality.*below"
Meaning: Data quality degradation
Action: Investigate data issues

Pattern: "memory.*high|exceeding"
Meaning: Memory pressure
Action: Monitor and consider scaling
```

### 9.4 Log Analysis Examples

**Find all emissions calculations for specific facility:**
```bash
greenlang logs search \
  --agent GL-010 \
  --pattern "calculation.*facility-001" \
  --time-range "last-24h" \
  --fields "timestamp,pollutant,value,method"
```

**Find all CEMS communication errors:**
```bash
greenlang logs search \
  --agent GL-010 \
  --pattern "CEMS-001|CEMS-002|connection.*failed" \
  --level error \
  --time-range "last-7d" \
  --group-by "source"
```

**Find all compliance determinations:**
```bash
greenlang logs search \
  --agent GL-010 \
  --log-type compliance \
  --time-range "last-24h" \
  --fields "timestamp,facility,pollutant,limit,actual,status"
```

---

## 10. Database Diagnostics

### 10.1 Database Health Checks

```bash
# Overall database health
greenlang db health --database gl010-emissions-db

# Connection pool status
greenlang db pool-status --database gl010-emissions-db

# Slow query analysis
greenlang db slow-queries \
  --database gl010-emissions-db \
  --threshold 1000ms \
  --last 24h

# Table statistics
greenlang db table-stats --database gl010-emissions-db

# Index usage
greenlang db index-usage --database gl010-emissions-db
```

### 10.2 Data Verification Queries

```bash
# Verify emissions data completeness
greenlang db query --database gl010-emissions-db --query "
SELECT
  facility_id,
  pollutant,
  DATE(timestamp) as date,
  COUNT(*) as records,
  COUNT(*) * 100.0 / (24 * 60) as completeness_pct
FROM emissions_data
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY facility_id, pollutant, DATE(timestamp)
HAVING COUNT(*) < 1440 * 0.9
ORDER BY date, facility_id, pollutant
"

# Check for data gaps
greenlang db query --database gl010-emissions-db --query "
SELECT
  facility_id,
  pollutant,
  gap_start,
  gap_end,
  EXTRACT(EPOCH FROM (gap_end - gap_start))/60 as gap_minutes
FROM (
  SELECT
    facility_id,
    pollutant,
    timestamp as gap_start,
    LEAD(timestamp) OVER (PARTITION BY facility_id, pollutant ORDER BY timestamp) as gap_end
  FROM emissions_data
  WHERE timestamp >= NOW() - INTERVAL '24 hours'
) gaps
WHERE gap_end - gap_start > INTERVAL '15 minutes'
"

# Verify calculation consistency
greenlang db query --database gl010-emissions-db --query "
SELECT
  e.facility_id,
  e.pollutant,
  e.timestamp,
  e.calculated_value,
  c.raw_value,
  ABS(e.calculated_value - c.raw_value) / c.raw_value * 100 as pct_diff
FROM emissions_data e
JOIN cems_data c ON e.facility_id = c.facility_id
  AND e.pollutant = c.pollutant
  AND e.timestamp = c.timestamp
WHERE e.timestamp >= NOW() - INTERVAL '24 hours'
  AND ABS(e.calculated_value - c.raw_value) / NULLIF(c.raw_value, 0) > 0.05
"
```

### 10.3 Database Maintenance

```bash
# Analyze tables for query optimization
greenlang db analyze --database gl010-emissions-db

# Vacuum tables to reclaim space
greenlang db vacuum --database gl010-emissions-db --table emissions_data

# Rebuild indexes
greenlang db reindex --database gl010-emissions-db --index idx_emissions_facility_time

# Archive old data
greenlang db archive \
  --database gl010-emissions-db \
  --table emissions_data \
  --older-than "3 years" \
  --destination archive-storage
```

---

## 11. Performance Troubleshooting

### 11.1 Performance Metrics

```bash
# Get performance summary
greenlang metrics summary --agent GL-010 --period last-1h

# CPU utilization
greenlang metrics query \
  --metric "cpu_usage" \
  --agent GL-010 \
  --time-range "last-4h" \
  --resolution "1m"

# Memory utilization
greenlang metrics query \
  --metric "memory_usage" \
  --agent GL-010 \
  --time-range "last-4h" \
  --resolution "1m"

# Request latency
greenlang metrics query \
  --metric "request_latency_p99" \
  --agent GL-010 \
  --time-range "last-4h" \
  --resolution "5m"

# Queue depth
greenlang metrics query \
  --metric "queue_depth" \
  --agent GL-010 \
  --time-range "last-4h" \
  --resolution "1m"
```

### 11.2 Performance Issues and Solutions

**High CPU Usage:**
```bash
# Profile CPU usage
greenlang profile cpu --agent GL-010 --duration 60s --output /tmp/cpu-profile.json

# Identify hot spots
greenlang analyze-profile /tmp/cpu-profile.json --top 10

# Common causes and solutions:
# - Complex calculations: Enable caching
# - High request rate: Scale horizontally
# - Inefficient queries: Optimize database queries
```

**High Memory Usage:**
```bash
# Profile memory
greenlang profile memory --agent GL-010 --duration 60s --output /tmp/mem-profile.json

# Check for leaks
greenlang analyze-profile /tmp/mem-profile.json --type heap

# Common causes and solutions:
# - Cache too large: Reduce cache size
# - Memory leak: Restart pod, report bug
# - Large data set: Implement pagination
```

**Slow Response Times:**
```bash
# Trace slow requests
greenlang trace slow-requests \
  --agent GL-010 \
  --threshold 1000ms \
  --last 1h

# Identify bottlenecks
greenlang analyze-traces \
  --agent GL-010 \
  --time-range "last-1h" \
  --show-bottlenecks

# Common causes and solutions:
# - Database slow: Optimize queries, add indexes
# - External service slow: Add timeouts, implement circuit breaker
# - Computation intensive: Add caching, optimize algorithms
```

**Queue Backlog:**
```bash
# Check queue status
greenlang queue status --agent GL-010

# Identify slow consumers
greenlang queue consumers --agent GL-010 --show-rates

# Common causes and solutions:
# - Consumer too slow: Scale consumers
# - Producer burst: Implement rate limiting
# - Dead letter queue growth: Investigate failed messages
```

### 11.3 Scaling Recommendations

| Issue | Threshold | Recommendation |
|-------|-----------|----------------|
| CPU > 80% sustained | 15 minutes | Scale horizontally or vertically |
| Memory > 85% | Any | Increase memory limit |
| Latency p99 > 2s | 5 minutes | Investigate bottleneck |
| Queue depth > 1000 | 5 minutes | Scale consumers |
| Error rate > 1% | 5 minutes | Investigate errors |

---

## 12. Appendices

### Appendix A: Troubleshooting Decision Tree

```
                           ISSUE DETECTED
                                │
                                ▼
              ┌────────────────────────────────────┐
              │         What type of issue?         │
              └────────────────────────────────────┘
                                │
    ┌───────────┬───────────┬───────────┬───────────┐
    │           │           │           │           │
    ▼           ▼           ▼           ▼           ▼
Emissions    CEMS      Compliance  Reporting   System
Calculation  Integration  Check     Issue      Issue
    │           │           │           │           │
    ▼           ▼           ▼           ▼           ▼
Section 3   Section 4   Section 5  Section 6  Section 11
```

### Appendix B: Quick Reference Card

**Most Common Issues:**

| Issue | Quick Command | Section |
|-------|---------------|---------|
| NOx mismatch | `greenlang emissions trace --pollutant NOx` | 3.1 |
| CEMS timeout | `greenlang cems test-connection --verbose` | 4.1 |
| Wrong limit | `greenlang compliance limits --show-all` | 5.1 |
| Report fail | `greenlang report validate --detail` | 6.1 |
| Slow perf | `greenlang metrics summary` | 11.1 |

### Appendix C: Related Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| Incident Response | ./INCIDENT_RESPONSE.md | Incident handling |
| Rollback Procedure | ./ROLLBACK_PROCEDURE.md | System rollback |
| Scaling Guide | ./SCALING_GUIDE.md | Capacity management |
| Maintenance Guide | ./MAINTENANCE.md | Routine maintenance |
| API Documentation | /docs/api/gl-010/ | API reference |

### Appendix D: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-26 | GL-TechWriter | Initial release |

### Appendix E: Feedback

Submit feedback to docs@greenlang.io with subject "GL-010 Troubleshooting Feedback"

---

**Document Classification:** Internal Use Only

**Next Review Date:** 2026-02-26
