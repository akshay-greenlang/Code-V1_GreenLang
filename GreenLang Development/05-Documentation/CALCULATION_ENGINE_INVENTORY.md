# GreenLang Calculation Engine Inventory

**Generated:** February 2, 2026
**Status:** Complete Zero-Hallucination Infrastructure

---

## Executive Summary

GreenLang contains comprehensive, regulatory-grade calculation engines with **zero-hallucination guarantees**. The inventory includes 50+ calculation engines covering Scope 1, 2, 3 emissions, CBAM compliance, ESRS/CSRD reporting, and industrial-specific calculations.

**Key Metrics:**
- 300+ emission factors from EPA, DEFRA, IPCC, IEA
- 520+ ESRS formulas (all deterministic)
- 15/15 Scope 3 categories implemented
- SHA-256 provenance for all calculations

---

## 1. Zero-Hallucination Guarantees

```
ZERO-HALLUCINATION GUARANTEE:
- NO LLM calls in calculation path
- 100% deterministic (same input → same output)
- Full provenance tracking with SHA-256 hashing
- Fail loudly on missing data or invalid inputs
```

### Determinism Features
- `DeterministicClock` - Reproducible timestamps
- `FinancialDecimal` - ROUND_HALF_UP with configurable precision
- AST-based formula parsing (no eval/exec)
- Bit-perfect reproducibility tests

---

## 2. Core Calculators

| Calculator | Location | Purpose |
|------------|----------|---------|
| **Core Calculator** | `greenlang/agents/calculation/emissions/core_calculator.py` | Base emission calculation engine |
| **CBAM Calculator** | `cbam-pack-mvp/src/cbam_pack/calculators/emissions_calculator.py` | CBAM compliance |
| **Formula Engine** | `agent_foundation/agents/calculator/formula_engine.py` | YAML-based formula library |
| **Scope 1 Calculator** | `greenlang/agents/calculation/emissions/scope1_calculator.py` | Stationary, mobile, fugitive |
| **Scope 2 Calculator** | `greenlang/agents/calculation/emissions/scope2_calculator.py` | Location/market-based |
| **Scope 3 Agent** | `GL-VCCI-Carbon-APP/.../calculator/agent.py` | All 15 categories |

---

## 3. Scope 3 Calculator (All 15 Categories)

| Cat | Name | Calculator | Status |
|-----|------|------------|--------|
| 1 | Purchased Goods & Services | Category1Calculator | ✅ |
| 2 | Capital Goods | Category2Calculator | ✅ |
| 3 | Fuel & Energy-Related | Category3Calculator | ✅ |
| 4 | Upstream Transportation | Category4Calculator (ISO 14083) | ✅ |
| 5 | Waste Generated | Category5Calculator | ✅ |
| 6 | Business Travel | Category6Calculator | ✅ |
| 7 | Employee Commuting | Category7Calculator | ✅ |
| 8 | Upstream Leased Assets | Category8Calculator | ✅ |
| 9 | Downstream Transportation | Category9Calculator | ✅ |
| 10 | Processing of Sold Products | Category10Calculator | ✅ |
| 11 | Use of Sold Products | Category11Calculator | ✅ |
| 12 | End-of-Life Treatment | Category12Calculator | ✅ |
| 13 | Downstream Leased Assets | Category13Calculator | ✅ |
| 14 | Franchises | Category14Calculator | ✅ |
| 15 | Investments | Category15Calculator (PCAF) | ✅ |

---

## 4. Three-Tier Calculation Waterfall

**GHG Protocol Hierarchy:**

| Tier | Data Type | DQI Score | Rating |
|------|-----------|-----------|--------|
| **Tier 1** | Supplier-specific PCF | 85 | Excellent |
| **Tier 2** | Average product EFs | 65 | Good |
| **Tier 3** | Spend-based proxy | 45 | Fair |

**Example - Category 1:**
```python
# Tier 1: emissions = quantity × supplier_pcf
# Tier 2: emissions = quantity × product_ef
# Tier 3: emissions = spend_usd × economic_intensity_ef
```

---

## 5. Emission Factor Registry

**Location:** `data/emission_factors_registry.yaml`

### Coverage

| Category | Examples |
|----------|----------|
| **Fuels** | Natural gas, LNG, coal, diesel, gasoline, propane, jet fuel, hydrogen |
| **Grids** | US eGRID, UK, Germany, France, China, India, Japan, Brazil, Canada, EU |
| **Processes** | Cement, steel, aluminum, refrigeration, waste, agriculture |
| **Travel** | Air (short/long haul), rail, hotel |
| **Water** | Municipal supply, wastewater |
| **Renewables** | Solar PV, wind, hydro, nuclear (lifecycle) |

### Standards
- GHG Protocol Corporate Standard
- ISO 14064-1:2018
- IPCC 2021 Guidelines
- EPA GHG Reporting Program
- IPCC AR6 GWP100 (default)

---

## 6. ESRS Formula Library

**Location:** `applications/GL-CSRD-APP/.../data/esrs_formulas.yaml`

**520+ Formulas covering:**
- E1: Climate Change (GHG, energy)
- E2: Pollution (NOx, SOx, PM)
- E3: Water and Marine
- E4: Biodiversity
- E5: Circular Economy
- S1-S4: Social metrics
- G1: Governance

**Example:**
```yaml
E1-1_scope1_total:
  formula: "scope1_stationary + scope1_mobile + scope1_process + scope1_fugitive"
  unit: "tCO2e"
  deterministic: true
  zero_hallucination: true

E1-8_ghg_intensity_revenue:
  formula: "E1-4 / (revenue / 1000000)"
  unit: "tCO2e per EUR million"
```

---

## 7. Industry-Specific Calculators

| Calculator | Industries |
|------------|------------|
| `CementEmissionCalculator` | Cement, clinker |
| `SteelEmissionCalculator` | Blast furnace, electric arc |
| `AluminumEmissionCalculator` | Primary/secondary smelting |
| `ChemicalEmissionCalculator` | Ammonia, hydrogen |
| `GlassEmissionCalculator` | Glass manufacturing |
| `PaperEmissionCalculator` | Pulp and paper |

---

## 8. Physics Calculators

| Calculator | Purpose |
|------------|---------|
| `PsychrometricCalculator` | Air properties, humidity |
| `HeatTransferCalculator` | LMTD, effectiveness |
| `ExergyCalculator` | Exergy analysis |
| `API530Calculator` | Pressure vessel design |
| `GUMUncertaintyCalculator` | Uncertainty propagation |

---

## 9. Unit Conversion System

**Supported Units:**

| Category | Units |
|----------|-------|
| **Energy** | Wh, kWh, MWh, GWh, J, kJ, MJ, GJ, BTU, MMBTU, therm |
| **Mass** | g, kg, t, kt, Mt, lb, oz, ton_us |
| **Volume** | mL, L, m³, gal_us, gal_uk, barrel |
| **Distance** | m, km, mi, nmi |
| **Emissions** | kg_co2e, t_co2e, kt_co2e, Mt_co2e |

---

## 10. Audit Trail System

**Location:** `GL-VCCI-Carbon-APP/.../reporting/compliance/audit_trail.py`

### Audit Package Structure
```python
{
    "audit_id": "SHA-256 hash",
    "generated_at": "ISO timestamp",
    "data_lineage": {...},
    "calculation_evidence": [...],
    "methodology_documentation": {...},
    "integrity_hashes": {
        "emissions_data_hash": "SHA-256",
        "calculations_hash": "SHA-256"
    }
}
```

### Hash Chain
1. Input data hash
2. Emission factor hash
3. Calculation step hashes
4. Final result hash
5. Timestamp

---

## 11. Performance Specifications

| Metric | Target | Implementation |
|--------|--------|----------------|
| Calculation latency | <5ms per record | Decimal arithmetic |
| Batch throughput | 100,000/hour | Parallel processing |
| Precision | 2-3 decimal places | ROUND_HALF_UP |
| Test coverage | 100% for formulas | Pytest + golden |
| Reproducibility | Bit-perfect | SHA-256 provenance |

---

## 12. Summary Statistics

| Category | Count |
|----------|-------|
| Calculation Engine Files | 50+ |
| Emission Factors | 300+ |
| ESRS Formulas | 520+ |
| Validation Rules | 100+ |
| Scope 3 Categories | 15/15 |
| Industry Calculators | 6+ |
| Unit Conversions | 40+ |
| Determinism Tests | 20+ |

---

## Key File Locations

| Component | Path |
|-----------|------|
| Core Calculator | `greenlang/agents/calculation/emissions/core_calculator.py` |
| CBAM Calculator | `cbam-pack-mvp/src/cbam_pack/calculators/emissions_calculator.py` |
| Scope 3 Agent | `applications/GL-VCCI-Carbon-APP/.../calculator/agent.py` |
| Formula Engine | `agent_foundation/agents/calculator/formula_engine.py` |
| Emission Factors | `data/emission_factors_registry.yaml` |
| ESRS Formulas | `applications/GL-CSRD-APP/.../esrs_formulas.yaml` |
| Validation Rules | `examples/validation_rules/calculation_validation.yaml` |
| Audit Trail | `GL-VCCI-Carbon-APP/.../audit_trail.py` |

---

*Document maintained by GreenLang Development Team*
*Last updated: February 2, 2026*
