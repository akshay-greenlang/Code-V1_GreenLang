# GL-CSRD-APP Complete Development Guide
## Part 2: Implementation & Operations

**Document Version:** 1.0
**Last Updated:** 2025-10-18
**Continuation of:** COMPLETE_DEVELOPMENT_GUIDE.md

---

## Table of Contents - Part 2

- **Part III:** Development Roadmap & Implementation Plan
- **Part IV:** Agent Ecosystem Integration (18-Agent Orchestration)
- **Part V:** Hands-On Implementation Guide
- **Part VI:** Operations, Deployment & Production
- **Part VII:** Business Strategy & Go-to-Market
- **Part VIII:** Appendices & Reference Materials

---

# PART III: DEVELOPMENT ROADMAP & IMPLEMENTATION PLAN

## 3.1 Overview: 4-Week Sprint to Production

**Current State:** 90% Complete (11,001 lines of production code)
**Target State:** 100% Production-Ready
**Timeline:** 20 working days (4 weeks)
**Phases Remaining:** 5 phases (5-8 + Production)

### **Phase Summary**

```
Week 1 (Days 1-5): Testing & Foundation
├── Phase 5: Comprehensive Testing Suite
├── Phase 6: Scripts & Utilities
└── Phase 7: Examples & Documentation

Week 2 (Days 6-10): Production Readiness & Agent Integration
├── Phase 8: Final Integration & Production Prep
└── Phase 9: GreenLang Agent Integration (14 agents)

Week 3 (Days 11-15): Domain Specialization
└── Phase 10: CSRD-Specific Agents (4 domain agents)

Week 4 (Days 16-20): Integration & Deployment
├── Phase 11: Full System Integration
└── Phase 12: Production Deployment
```

---

## 3.2 Week 1: Testing & Foundation (Days 1-5)

### **Day 1: CalculatorAgent Testing (CRITICAL)**

**Priority:** CRITICAL - Zero-hallucination guarantee must be verified

**Objective:** Achieve 100% test coverage for CalculatorAgent

**Tasks:**
1. Create comprehensive test suite structure
2. Test all 520+ formulas from `data/esrs_formulas.yaml`
3. Verify GHG Protocol calculations (Scope 1/2/3)
4. Test emission factor database lookups
5. Verify reproducibility (same inputs → same outputs)
6. Test edge cases (missing data, invalid formulas, boundary conditions)

**Deliverables:**

```python
# tests/unit/test_calculator_agent.py

import pytest
from agents.calculator_agent import CalculatorAgent
from pathlib import Path
import yaml
import json

@pytest.fixture
def calculator():
    """Initialize CalculatorAgent with test configuration"""
    return CalculatorAgent(
        config_path="config/csrd_config.yaml",
        formulas_path="data/esrs_formulas.yaml",
        emission_factors_path="data/emission_factors.json"
    )

@pytest.fixture
def sample_fuel_data():
    """Sample fuel consumption data for testing"""
    return {
        "natural_gas_m3": 45000,
        "diesel_L": 2500,
        "electricity_MWh": 1500,
        "reporting_year": 2024,
        "country": "Germany"
    }

class TestGHGProtocol:
    """Test GHG Protocol calculations (Scope 1/2/3)"""

    def test_scope1_emissions_natural_gas(self, calculator, sample_fuel_data):
        """Test Scope 1 emissions calculation for natural gas"""
        result = calculator.calculate_scope1_emissions({
            'natural_gas': sample_fuel_data['natural_gas_m3']
        })

        # Expected: 45000 m³ × 0.2016 tCO2e/m³ = 9072.0 tCO2e
        assert result['value'] == 9072.0
        assert result['unit'] == 'tCO2e'
        assert result['metric_code'] == 'E1-1'
        assert result['provenance'] is not None
        assert 'natural_gas' in result['provenance']['inputs']

    def test_scope1_emissions_diesel(self, calculator, sample_fuel_data):
        """Test Scope 1 emissions calculation for diesel"""
        result = calculator.calculate_scope1_emissions({
            'diesel': sample_fuel_data['diesel_L']
        })

        # Expected: 2500 L × 2.687 tCO2e/L = 6717.5 tCO2e
        assert result['value'] == 6717.5
        assert result['unit'] == 'tCO2e'

    def test_scope2_emissions_electricity(self, calculator, sample_fuel_data):
        """Test Scope 2 emissions (location-based)"""
        result = calculator.calculate_scope2_emissions({
            'electricity_MWh': sample_fuel_data['electricity_MWh'],
            'country': sample_fuel_data['country']
        })

        # Expected: 1500 MWh × 0.385 tCO2e/MWh (Germany grid 2024) = 577.5 tCO2e
        assert result['value'] == 577.5
        assert result['unit'] == 'tCO2e'
        assert result['metric_code'] == 'E1-2'
        assert result['calculation_method'] == 'location-based'

    def test_combined_scope1_scope2(self, calculator, sample_fuel_data):
        """Test combined Scope 1 + Scope 2 emissions"""
        result = calculator.calculate_combined_emissions({
            'natural_gas_m3': sample_fuel_data['natural_gas_m3'],
            'diesel_L': sample_fuel_data['diesel_L'],
            'electricity_MWh': sample_fuel_data['electricity_MWh'],
            'country': sample_fuel_data['country']
        })

        # Expected: 9072.0 + 6717.5 + 577.5 = 16367.0 tCO2e
        assert result['scope1']['value'] == 15789.5
        assert result['scope2']['value'] == 577.5
        assert result['total']['value'] == 16367.0

class TestFormulaEngine:
    """Test the formula engine with all 520+ formulas"""

    def test_load_all_formulas(self, calculator):
        """Verify all formulas load correctly from YAML"""
        formulas = calculator.load_formulas()

        assert len(formulas) >= 520, "Should have 520+ formulas"
        assert 'E1-1' in formulas, "Should include Scope 1 emissions"
        assert 'E1-4' in formulas, "Should include total energy"
        assert 'S1-1' in formulas, "Should include employee metrics"

    def test_formula_structure_validation(self, calculator):
        """Verify each formula has required fields"""
        formulas = calculator.load_formulas()

        for code, formula in formulas.items():
            assert 'metric_name' in formula
            assert 'formula' in formula
            assert 'formula_type' in formula
            assert 'inputs' in formula
            assert 'calculation_steps' in formula
            assert 'output_unit' in formula
            assert 'authoritative_source' in formula

    def test_simple_sum_formula(self, calculator):
        """Test simple sum formula type (e.g., S1-1 Total Employees)"""
        result = calculator.calculate_metric(
            metric_code='S1-1',
            input_data={
                'full_time': 250,
                'part_time': 50,
                'contractors': 25
            }
        )

        # Expected: 250 + 50 + 25 = 325 FTE
        assert result['value'] == 325
        assert result['unit'] == 'FTE'

    def test_multiplication_sum_formula(self, calculator):
        """Test multiplication + sum formula type (e.g., E1-1)"""
        result = calculator.calculate_metric(
            metric_code='E1-1',
            input_data={
                'fuels': [
                    {'type': 'natural_gas', 'consumption': 45000, 'unit': 'm3'},
                    {'type': 'diesel', 'consumption': 2500, 'unit': 'L'}
                ]
            }
        )

        # Expected: (45000 × 0.2016) + (2500 × 2.687) = 15789.5 tCO2e
        assert result['value'] == 15789.5

    def test_weighted_average_formula(self, calculator):
        """Test weighted average formula type"""
        result = calculator.calculate_metric(
            metric_code='E1-3',  # GHG Intensity
            input_data={
                'total_ghg_emissions': 16367.0,
                'revenue_million_eur': 50.0
            }
        )

        # Expected: 16367.0 / 50.0 = 327.34 tCO2e/M€
        assert abs(result['value'] - 327.34) < 0.01
        assert result['unit'] == 'tCO2e/M€'

class TestReproducibility:
    """CRITICAL: Test zero-hallucination reproducibility guarantee"""

    def test_identical_outputs_for_identical_inputs(self, calculator, sample_fuel_data):
        """Verify same inputs produce EXACTLY the same outputs"""
        # Run 1
        result1 = calculator.calculate_metric(
            metric_code='E1-1',
            input_data=sample_fuel_data
        )

        # Run 2 (identical inputs)
        result2 = calculator.calculate_metric(
            metric_code='E1-1',
            input_data=sample_fuel_data
        )

        # Verify byte-identical results
        assert result1['value'] == result2['value']
        assert result1['provenance_hash'] == result2['provenance_hash']
        assert result1['calculation_timestamp']  # Timestamp will differ, but hash should match

    def test_hash_stability_across_runs(self, calculator):
        """Verify SHA-256 hash is stable for identical inputs"""
        input_data = {'natural_gas_m3': 45000}

        hashes = []
        for i in range(10):
            result = calculator.calculate_metric('E1-1', input_data)
            hashes.append(result['provenance_hash'])

        # All hashes should be identical
        assert len(set(hashes)) == 1, "All hashes should be identical"

    def test_no_floating_point_drift(self, calculator):
        """Verify no accumulation of floating-point errors"""
        # Calculate emissions 1000 times
        input_data = {'natural_gas_m3': 45000}

        results = []
        for i in range(1000):
            result = calculator.calculate_metric('E1-1', input_data)
            results.append(result['value'])

        # All results should be EXACTLY identical (no drift)
        assert len(set(results)) == 1, "No floating-point drift allowed"
        assert results[0] == 9072.0

class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_missing_emission_factor(self, calculator):
        """Test behavior when emission factor not found"""
        with pytest.raises(ValueError, match="Emission factor not found"):
            calculator.calculate_metric(
                metric_code='E1-1',
                input_data={'unknown_fuel': 1000}
            )

    def test_zero_consumption(self, calculator):
        """Test zero fuel consumption"""
        result = calculator.calculate_metric(
            metric_code='E1-1',
            input_data={'natural_gas_m3': 0}
        )

        assert result['value'] == 0.0
        assert result['unit'] == 'tCO2e'

    def test_negative_consumption_rejected(self, calculator):
        """Test that negative values are rejected"""
        with pytest.raises(ValueError, match="Consumption cannot be negative"):
            calculator.calculate_metric(
                metric_code='E1-1',
                input_data={'natural_gas_m3': -1000}
            )

    def test_missing_required_input(self, calculator):
        """Test missing required input field"""
        with pytest.raises(ValueError, match="Missing required input"):
            calculator.calculate_metric(
                metric_code='E1-1',
                input_data={}
            )

    def test_invalid_unit_conversion(self, calculator):
        """Test invalid unit conversion"""
        with pytest.raises(ValueError, match="Cannot convert"):
            calculator.calculate_metric(
                metric_code='E1-1',
                input_data={'natural_gas': 45000, 'unit': 'invalid_unit'}
            )

class TestProvenanceTracking:
    """Test complete provenance tracking"""

    def test_provenance_record_completeness(self, calculator, sample_fuel_data):
        """Verify provenance record contains all required fields"""
        result = calculator.calculate_metric('E1-1', sample_fuel_data)

        prov = result['provenance']
        assert 'metric_code' in prov
        assert 'metric_name' in prov
        assert 'calculation_timestamp' in prov
        assert 'formula' in prov
        assert 'inputs' in prov
        assert 'emission_factors' in prov
        assert 'calculation_steps' in prov
        assert 'provenance_hash' in prov
        assert 'environment' in prov

    def test_provenance_hash_format(self, calculator, sample_fuel_data):
        """Verify provenance hash is valid SHA-256"""
        result = calculator.calculate_metric('E1-1', sample_fuel_data)

        hash_value = result['provenance_hash']
        assert hash_value.startswith('sha256:')
        assert len(hash_value) == 71  # 'sha256:' + 64 hex chars

    def test_environment_snapshot(self, calculator, sample_fuel_data):
        """Verify environment snapshot captured"""
        result = calculator.calculate_metric('E1-1', sample_fuel_data)

        env = result['provenance']['environment']
        assert 'python_version' in env
        assert 'platform' in env
        assert 'calculator_version' in env
        assert 'emission_factors_db_version' in env

class TestPerformance:
    """Test performance requirements"""

    def test_calculation_speed(self, calculator, sample_fuel_data):
        """Verify calculation completes in <100ms"""
        import time

        start = time.time()
        result = calculator.calculate_metric('E1-1', sample_fuel_data)
        duration = time.time() - start

        assert duration < 0.1, f"Calculation took {duration:.3f}s, should be <0.1s"

    def test_batch_calculation_throughput(self, calculator):
        """Verify can process 500+ metrics/second"""
        import time

        # Prepare 1000 calculations
        input_data = {'natural_gas_m3': 45000}

        start = time.time()
        for i in range(1000):
            calculator.calculate_metric('E1-1', input_data)
        duration = time.time() - start

        throughput = 1000 / duration
        assert throughput >= 500, f"Throughput {throughput:.0f}/s, should be >=500/s"

# Additional test classes for all 520+ formulas

class TestESRS_E1_Climate:
    """Test all ESRS E1 (Climate Change) metrics"""

    def test_E1_1_scope1_emissions(self, calculator):
        """Test E1-1: Scope 1 GHG Emissions"""
        # Already tested above
        pass

    def test_E1_2_scope2_emissions(self, calculator):
        """Test E1-2: Scope 2 GHG Emissions"""
        # Already tested above
        pass

    def test_E1_3_ghg_intensity(self, calculator):
        """Test E1-3: GHG Intensity"""
        # Already tested above
        pass

    def test_E1_4_total_energy_consumption(self, calculator):
        """Test E1-4: Total Energy Consumption"""
        result = calculator.calculate_metric(
            metric_code='E1-4',
            input_data={
                'electricity_MWh': 1500,
                'natural_gas_m3': 45000,
                'diesel_L': 2500
            }
        )

        # Expected:
        # - Electricity: 1500 MWh × 3.6 = 5400 GJ
        # - Natural gas: 45000 m³ × 0.0378 = 1701 GJ
        # - Diesel: 2500 L × 0.0356 = 89 GJ
        # - Total: 5400 + 1701 + 89 = 7190 GJ
        assert result['value'] == 7190.0
        assert result['unit'] == 'GJ'

    def test_E1_5_renewable_energy_percentage(self, calculator):
        """Test E1-5: Renewable Energy Percentage"""
        result = calculator.calculate_metric(
            metric_code='E1-5',
            input_data={
                'renewable_energy_GJ': 2876,
                'total_energy_GJ': 7190
            }
        )

        # Expected: (2876 / 7190) × 100 = 40.0%
        assert abs(result['value'] - 40.0) < 0.1
        assert result['unit'] == '%'

    # ... Continue with all E1 metrics (E1-6 through E1-9)

class TestESRS_E2_Pollution:
    """Test all ESRS E2 (Pollution) metrics"""

    def test_E2_1_emissions_to_air(self, calculator):
        """Test E2-1: Emissions to Air"""
        result = calculator.calculate_metric(
            metric_code='E2-1',
            input_data={
                'NOx_tonnes': 2.5,
                'SOx_tonnes': 1.2,
                'PM_tonnes': 0.8
            }
        )

        # Expected: 2.5 + 1.2 + 0.8 = 4.5 tonnes
        assert result['value'] == 4.5
        assert result['unit'] == 'tonnes'

    # ... Continue with all E2 metrics

class TestESRS_E3_Water:
    """Test all ESRS E3 (Water & Marine Resources) metrics"""
    pass

class TestESRS_E4_Biodiversity:
    """Test all ESRS E4 (Biodiversity & Ecosystems) metrics"""
    pass

class TestESRS_E5_CircularEconomy:
    """Test all ESRS E5 (Circular Economy) metrics"""
    pass

class TestESRS_S1_Workforce:
    """Test all ESRS S1 (Own Workforce) metrics"""

    def test_S1_1_total_employees(self, calculator):
        """Test S1-1: Total Employees"""
        # Already tested above
        pass

    # ... Continue with all S1 metrics

class TestESRS_S2_Workers:
    """Test all ESRS S2 (Workers in Value Chain) metrics"""
    pass

class TestESRS_S3_Communities:
    """Test all ESRS S3 (Affected Communities) metrics"""
    pass

class TestESRS_S4_Consumers:
    """Test all ESRS S4 (Consumers & End-Users) metrics"""
    pass

class TestESRS_G1_Governance:
    """Test all ESRS G1 (Business Conduct) metrics"""
    pass

# Run all tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=agents.calculator_agent", "--cov-report=html"])
```

**Success Criteria:**
- ✅ 100% test coverage for CalculatorAgent
- ✅ All 520+ formulas tested
- ✅ Zero-hallucination reproducibility verified
- ✅ Performance targets met (<100ms per calculation, 500+ metrics/sec)
- ✅ Edge cases handled correctly

**Time Estimate:** 6-8 hours

---

### **Day 2: Core Agent Tests (IntakeAgent, AuditAgent)**

**Objective:** Achieve 90-95% test coverage for IntakeAgent and AuditAgent

**IntakeAgent Tests:**

```python
# tests/unit/test_intake_agent.py

import pytest
from agents.intake_agent import IntakeAgent
import pandas as pd
from pathlib import Path

@pytest.fixture
def intake_agent():
    """Initialize IntakeAgent"""
    return IntakeAgent(config_path="config/csrd_config.yaml")

class TestDataIngestion:
    """Test multi-format data ingestion"""

    def test_ingest_csv(self, intake_agent, tmp_path):
        """Test CSV file ingestion"""
        # Create sample CSV
        csv_path = tmp_path / "test_data.csv"
        df = pd.DataFrame({
            'metric_code': ['E1-1', 'E1-2'],
            'value': [9072.0, 577.5],
            'unit': ['tCO2e', 'tCO2e'],
            'reporting_year': [2024, 2024]
        })
        df.to_csv(csv_path, index=False)

        # Ingest
        result = intake_agent.ingest(csv_path)

        assert result['status'] == 'success'
        assert result['records_ingested'] == 2
        assert 'data_quality_score' in result

    def test_ingest_excel(self, intake_agent, tmp_path):
        """Test Excel file ingestion"""
        excel_path = tmp_path / "test_data.xlsx"
        df = pd.DataFrame({
            'metric_code': ['E1-1', 'E1-2'],
            'value': [9072.0, 577.5],
            'unit': ['tCO2e', 'tCO2e']
        })
        df.to_excel(excel_path, index=False)

        result = intake_agent.ingest(excel_path)

        assert result['status'] == 'success'
        assert result['records_ingested'] == 2

    def test_ingest_json(self, intake_agent, tmp_path):
        """Test JSON file ingestion"""
        json_path = tmp_path / "test_data.json"
        data = [
            {'metric_code': 'E1-1', 'value': 9072.0, 'unit': 'tCO2e'},
            {'metric_code': 'E1-2', 'value': 577.5, 'unit': 'tCO2e'}
        ]

        import json
        with open(json_path, 'w') as f:
            json.dump(data, f)

        result = intake_agent.ingest(json_path)

        assert result['status'] == 'success'
        assert result['records_ingested'] == 2

class TestSchemaValidation:
    """Test JSON schema validation"""

    def test_valid_schema(self, intake_agent):
        """Test data that passes schema validation"""
        data = {
            'metric_code': 'E1-1',
            'value': 9072.0,
            'unit': 'tCO2e',
            'reporting_year': 2024
        }

        result = intake_agent.validate_schema(data, schema_name='esg_data')

        assert result['valid'] is True
        assert len(result['errors']) == 0

    def test_invalid_schema(self, intake_agent):
        """Test data that fails schema validation"""
        data = {
            'metric_code': 'E1-1',
            'value': 'invalid',  # Should be numeric
            'unit': 'tCO2e'
        }

        result = intake_agent.validate_schema(data, schema_name='esg_data')

        assert result['valid'] is False
        assert len(result['errors']) > 0
        assert 'value' in result['errors'][0]

class TestDataQualityAssessment:
    """Test data quality scoring"""

    def test_high_quality_data(self, intake_agent):
        """Test data quality assessment for high-quality data"""
        data = pd.DataFrame({
            'metric_code': ['E1-1', 'E1-2', 'E1-3'],
            'value': [9072.0, 577.5, 327.34],
            'unit': ['tCO2e', 'tCO2e', 'tCO2e/M€'],
            'reporting_year': [2024, 2024, 2024],
            'source': ['invoice', 'meter', 'calculation']
        })

        quality_score = intake_agent.assess_data_quality(data)

        assert quality_score['overall_score'] >= 0.9
        assert quality_score['completeness'] >= 0.95
        assert quality_score['accuracy_indicators'] >= 0.9

    def test_low_quality_data(self, intake_agent):
        """Test data quality assessment for low-quality data"""
        data = pd.DataFrame({
            'metric_code': ['E1-1', 'E1-2', None],
            'value': [9072.0, None, 327.34],
            'unit': ['tCO2e', 'tCO2e', None]
        })

        quality_score = intake_agent.assess_data_quality(data)

        assert quality_score['overall_score'] < 0.7
        assert quality_score['completeness'] < 0.8

class TestThroughputPerformance:
    """Test 1000+ records/second ingestion"""

    def test_large_dataset_ingestion(self, intake_agent, tmp_path):
        """Test ingestion of 10,000 records"""
        import time

        # Create large dataset
        csv_path = tmp_path / "large_data.csv"
        data = {
            'metric_code': ['E1-1'] * 10000,
            'value': [9072.0] * 10000,
            'unit': ['tCO2e'] * 10000,
            'reporting_year': [2024] * 10000
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

        start = time.time()
        result = intake_agent.ingest(csv_path)
        duration = time.time() - start

        throughput = 10000 / duration

        assert result['records_ingested'] == 10000
        assert throughput >= 1000, f"Throughput {throughput:.0f}/s, should be >=1000/s"
```

**AuditAgent Tests:**

```python
# tests/unit/test_audit_agent.py

import pytest
from agents.audit_agent import AuditAgent

@pytest.fixture
def audit_agent():
    """Initialize AuditAgent"""
    return AuditAgent(
        config_path="config/csrd_config.yaml",
        rules_path="rules/esrs_compliance_rules.yaml"
    )

class TestComplianceRules:
    """Test 215+ compliance rules"""

    def test_load_all_rules(self, audit_agent):
        """Verify all 215+ rules load correctly"""
        rules = audit_agent.load_rules()

        assert len(rules) >= 215
        assert 'ESRS-E1-R1' in rules
        assert 'ESRS-S1-R1' in rules

    def test_rule_execution_success(self, audit_agent):
        """Test successful rule execution"""
        data = {
            'metric_code': 'E1-1',
            'value': 9072.0,
            'unit': 'tCO2e',
            'reporting_year': 2024
        }

        result = audit_agent.execute_rule('ESRS-E1-R1', data)

        assert result['status'] == 'pass'
        assert result['rule_id'] == 'ESRS-E1-R1'

    def test_rule_execution_failure(self, audit_agent):
        """Test failed rule execution"""
        data = {
            'metric_code': 'E1-1',
            'value': -100,  # Invalid negative emissions
            'unit': 'tCO2e'
        }

        result = audit_agent.execute_rule('ESRS-E1-R1', data)

        assert result['status'] == 'fail'
        assert 'error_message' in result

class TestAuditPackageGeneration:
    """Test audit package creation"""

    def test_generate_audit_package(self, audit_agent, tmp_path):
        """Test complete audit package generation"""
        report_data = {
            'company_name': 'Test Company',
            'reporting_year': 2024,
            'metrics': [
                {'metric_code': 'E1-1', 'value': 9072.0}
            ]
        }

        package_path = audit_agent.generate_audit_package(
            report_data,
            output_dir=tmp_path
        )

        assert package_path.exists()
        assert package_path.suffix == '.zip'

        # Verify package contents
        import zipfile
        with zipfile.ZipFile(package_path, 'r') as zf:
            files = zf.namelist()
            assert 'audit_report.json' in files
            assert 'compliance_results.json' in files
            assert 'provenance_records/' in str(files)
```

**Success Criteria:**
- ✅ IntakeAgent: 90% test coverage
- ✅ AuditAgent: 95% test coverage
- ✅ All compliance rules tested
- ✅ Performance targets met

**Time Estimate:** 8 hours

---

### **Day 3: Integration Agent Tests (AggregatorAgent, MaterialityAgent)**

**Objective:** Achieve 80-90% test coverage for AggregatorAgent and MaterialityAgent

**AggregatorAgent Tests:**

```python
# tests/unit/test_aggregator_agent.py

import pytest
from agents.aggregator_agent import AggregatorAgent

@pytest.fixture
def aggregator():
    """Initialize AggregatorAgent"""
    return AggregatorAgent(
        config_path="config/csrd_config.yaml",
        mappings_path="data/framework_mappings.json"
    )

class TestFrameworkMappings:
    """Test TCFD/GRI/SASB → ESRS mappings"""

    def test_tcfd_to_esrs_mapping(self, aggregator):
        """Test TCFD → ESRS conversion"""
        tcfd_data = {
            'metric_name': 'Scope 1 Emissions',
            'value': 9072.0,
            'framework': 'TCFD'
        }

        esrs_data = aggregator.map_to_esrs(tcfd_data, source_framework='TCFD')

        assert esrs_data['metric_code'] == 'E1-1'
        assert esrs_data['value'] == 9072.0
        assert esrs_data['unit'] == 'tCO2e'

    def test_gri_to_esrs_mapping(self, aggregator):
        """Test GRI → ESRS conversion"""
        gri_data = {
            'metric_name': 'GRI 305-1',
            'value': 9072.0,
            'framework': 'GRI'
        }

        esrs_data = aggregator.map_to_esrs(gri_data, source_framework='GRI')

        assert esrs_data['metric_code'] == 'E1-1'

    def test_sasb_to_esrs_mapping(self, aggregator):
        """Test SASB → ESRS conversion"""
        sasb_data = {
            'metric_name': 'EM-IS-110a.1',  # SASB Steel industry emissions
            'value': 9072.0,
            'framework': 'SASB'
        }

        esrs_data = aggregator.map_to_esrs(sasb_data, source_framework='SASB')

        assert esrs_data['metric_code'] == 'E1-1'

class TestTimeSeriesAnalysis:
    """Test time-series analysis and trending"""

    def test_year_over_year_comparison(self, aggregator):
        """Test YoY emissions trend analysis"""
        historical_data = {
            2022: {'E1-1': 10000.0},
            2023: {'E1-1': 9500.0},
            2024: {'E1-1': 9072.0}
        }

        trend = aggregator.analyze_trend('E1-1', historical_data)

        assert trend['direction'] == 'decreasing'
        assert trend['yoy_change_2023_2024'] == -4.5  # -4.5% reduction
        assert trend['total_reduction_pct'] == -9.28  # -9.28% from 2022 to 2024

    def test_target_tracking(self, aggregator):
        """Test emissions reduction target tracking"""
        target = {
            'baseline_year': 2022,
            'baseline_value': 10000.0,
            'target_year': 2030,
            'target_reduction_pct': 50.0  # 50% reduction target
        }

        current_value = 9072.0
        current_year = 2024

        progress = aggregator.track_target_progress(
            target, current_value, current_year
        )

        assert progress['on_track'] is True
        assert progress['progress_pct'] > 0
        assert progress['remaining_reduction'] < target['target_reduction_pct']

class TestBenchmarkComparisons:
    """Test industry benchmark comparisons"""

    def test_industry_benchmark_comparison(self, aggregator):
        """Test comparison against industry benchmarks"""
        company_data = {
            'E1-3': 327.34,  # GHG intensity: 327.34 tCO2e/M€
            'industry': 'Manufacturing',
            'revenue_million_eur': 50.0
        }

        benchmark = aggregator.compare_to_industry_benchmark(company_data)

        assert 'industry_median' in benchmark
        assert 'industry_quartile' in benchmark
        assert 'percentile_rank' in benchmark
        assert benchmark['comparison'] in ['above', 'at', 'below']
```

**MaterialityAgent Tests (with Mock LLM):**

```python
# tests/unit/test_materiality_agent.py

import pytest
from agents.materiality_agent import MaterialityAgent
from unittest.mock import Mock, patch

@pytest.fixture
def materiality_agent():
    """Initialize MaterialityAgent with mocked LLM"""
    agent = MaterialityAgent(config_path="config/csrd_config.yaml")
    return agent

class TestMaterialityScoring:
    """Test materiality assessment scoring"""

    @patch('agents.materiality_agent.AnthropicClient')
    def test_impact_materiality_assessment(self, mock_llm, materiality_agent):
        """Test impact materiality scoring with mocked LLM"""
        # Mock LLM response
        mock_llm.return_value.complete.return_value = {
            'score': 4.2,
            'reasoning': 'High impact on climate due to manufacturing emissions',
            'confidence': 0.85
        }

        result = materiality_agent.assess_impact_materiality(
            topic='E1',  # Climate Change
            company_profile={
                'industry': 'Manufacturing',
                'revenue_million_eur': 50.0,
                'employees': 325
            }
        )

        assert result['score'] >= 1.0 and result['score'] <= 5.0
        assert result['requires_expert_review'] is True  # AI suggestions always flagged
        assert 'reasoning' in result

    @patch('agents.materiality_agent.AnthropicClient')
    def test_financial_materiality_assessment(self, mock_llm, materiality_agent):
        """Test financial materiality scoring"""
        mock_llm.return_value.complete.return_value = {
            'score': 3.8,
            'reasoning': 'Carbon pricing risk moderate',
            'confidence': 0.75
        }

        result = materiality_agent.assess_financial_materiality(
            topic='E1',
            company_profile={
                'industry': 'Manufacturing',
                'carbon_price_exposure': 'medium'
            }
        )

        assert result['score'] >= 1.0 and result['score'] <= 5.0
        assert result['requires_expert_review'] is True

class TestRAGRegulatory Intelligence:
    """Test RAG-based regulatory guidance retrieval"""

    @patch('agents.materiality_agent.ChromaDB')
    def test_retrieve_esrs_guidance(self, mock_vector_db, materiality_agent):
        """Test RAG retrieval of ESRS guidance"""
        # Mock vector DB response
        mock_vector_db.return_value.query.return_value = [
            {
                'text': 'ESRS E1 requires disclosure of Scope 1, 2, and 3 emissions...',
                'source': 'ESRS E1 Climate Change',
                'page': 12
            }
        ]

        guidance = materiality_agent.retrieve_regulatory_guidance(
            query='What are E1 disclosure requirements?'
        )

        assert len(guidance) > 0
        assert 'ESRS E1' in guidance[0]['text']
        assert guidance[0]['source'] == 'ESRS E1 Climate Change'

class TestExpertReviewFlags:
    """Test that all AI outputs are flagged for expert review"""

    def test_all_assessments_flagged(self, materiality_agent):
        """Verify all materiality assessments require expert review"""
        with patch('agents.materiality_agent.AnthropicClient') as mock_llm:
            mock_llm.return_value.complete.return_value = {
                'score': 4.0,
                'reasoning': 'Test',
                'confidence': 0.9
            }

            result = materiality_agent.assess_impact_materiality(
                topic='E1',
                company_profile={'industry': 'Manufacturing'}
            )

            assert result['requires_expert_review'] is True
            assert result['ai_generated'] is True
```

**Success Criteria:**
- ✅ AggregatorAgent: 90% test coverage
- ✅ MaterialityAgent: 80% test coverage (lower due to AI mocking complexity)
- ✅ Framework mappings verified (350+ mappings)
- ✅ Expert review flags working correctly

**Time Estimate:** 8 hours

---

### **Day 4: ReportingAgent Tests + Integration Tests**

**Objective:** Complete ReportingAgent tests (85% coverage) and begin integration tests

**ReportingAgent Tests:**

```python
# tests/unit/test_reporting_agent.py

import pytest
from agents.reporting_agent import ReportingAgent
from lxml import etree

@pytest.fixture
def reporting_agent():
    """Initialize ReportingAgent"""
    return ReportingAgent(config_path="config/csrd_config.yaml")

class TestXBRLGeneration:
    """Test XBRL generation"""

    def test_generate_xbrl_document(self, reporting_agent):
        """Test XBRL document generation"""
        report_data = {
            'company_name': 'Test Company',
            'reporting_year': 2024,
            'metrics': [
                {'metric_code': 'E1-1', 'value': 9072.0, 'unit': 'tCO2e'}
            ]
        }

        xbrl_doc = reporting_agent.generate_xbrl(report_data)

        # Verify XBRL structure
        root = etree.fromstring(xbrl_doc.encode())
        assert root.tag.endswith('xbrl')

        # Verify namespace declarations
        namespaces = root.nsmap
        assert 'esrs' in namespaces
        assert 'xbrli' in namespaces

    def test_xbrl_validation(self, reporting_agent):
        """Test XBRL validation against schema"""
        report_data = {
            'company_name': 'Test Company',
            'reporting_year': 2024,
            'metrics': [{'metric_code': 'E1-1', 'value': 9072.0}]
        }

        xbrl_doc = reporting_agent.generate_xbrl(report_data)
        validation = reporting_agent.validate_xbrl(xbrl_doc)

        assert validation['valid'] is True
        assert len(validation['errors']) == 0

class TestIXBRLGeneration:
    """Test iXBRL (Inline XBRL) generation"""

    def test_generate_ixbrl_html(self, reporting_agent):
        """Test iXBRL HTML generation"""
        report_data = {
            'company_name': 'Test Company',
            'reporting_year': 2024,
            'metrics': [{'metric_code': 'E1-1', 'value': 9072.0}]
        }

        ixbrl_html = reporting_agent.generate_ixbrl(report_data)

        # Verify HTML structure
        assert '<html' in ixbrl_html
        assert 'ix:nonNumeric' in ixbrl_html or 'ix:nonFraction' in ixbrl_html
        assert 'Test Company' in ixbrl_html

class TestESEFPackage:
    """Test ESEF package creation"""

    def test_create_esef_package(self, reporting_agent, tmp_path):
        """Test ESEF-compliant package creation"""
        report_data = {
            'company_name': 'Test Company',
            'reporting_year': 2024,
            'metrics': [{'metric_code': 'E1-1', 'value': 9072.0}]
        }

        package_path = reporting_agent.create_esef_package(
            report_data,
            output_dir=tmp_path
        )

        assert package_path.exists()
        assert package_path.suffix == '.zip'

        # Verify package structure
        import zipfile
        with zipfile.ZipFile(package_path, 'r') as zf:
            files = zf.namelist()
            assert any('META-INF' in f for f in files)
            assert any('.xhtml' in f for f in files)
            assert any('reports/' in f for f in files)

class TestNarrativeGeneration:
    """Test AI-generated narrative sections"""

    @patch('agents.reporting_agent.AnthropicClient')
    def test_generate_management_commentary(self, mock_llm, reporting_agent):
        """Test AI-generated management commentary"""
        mock_llm.return_value.complete.return_value = {
            'narrative': 'The company achieved a 9.3% reduction in Scope 1 emissions...',
            'confidence': 0.85
        }

        report_data = {
            'company_name': 'Test Company',
            'metrics': [
                {'metric_code': 'E1-1', 'value': 9072.0, 'previous_year': 10000.0}
            ]
        }

        narrative = reporting_agent.generate_management_commentary(report_data)

        assert narrative['requires_expert_review'] is True
        assert narrative['ai_generated'] is True
        assert len(narrative['text']) > 0
```

**Integration Tests:**

```python
# tests/integration/test_pipeline_integration.py

import pytest
from csrd_pipeline import CSRDPipeline
import pandas as pd

@pytest.fixture
def pipeline():
    """Initialize full CSRD pipeline"""
    return CSRDPipeline(config_path="config/csrd_config.yaml")

class TestEndToEndPipeline:
    """Test complete pipeline execution"""

    def test_full_pipeline_execution(self, pipeline, tmp_path):
        """Test end-to-end pipeline from CSV → XBRL report"""
        # Prepare input data
        input_csv = tmp_path / "input_data.csv"
        df = pd.DataFrame({
            'metric_code': ['E1-1', 'E1-2'],
            'value': [9072.0, 577.5],
            'unit': ['tCO2e', 'tCO2e'],
            'reporting_year': [2024, 2024]
        })
        df.to_csv(input_csv, index=False)

        # Run pipeline
        result = pipeline.run(
            input_path=input_csv,
            output_dir=tmp_path,
            generate_xbrl=True
        )

        # Verify all stages completed
        assert result['intake']['status'] == 'success'
        assert result['calculation']['status'] == 'success'
        assert result['audit']['status'] == 'success'
        assert result['reporting']['status'] == 'success'

        # Verify outputs
        assert (tmp_path / 'csrd_report.xbrl').exists()
        assert (tmp_path / 'audit_package.zip').exists()

    def test_pipeline_error_handling(self, pipeline, tmp_path):
        """Test pipeline handles errors gracefully"""
        # Prepare invalid input data
        input_csv = tmp_path / "invalid_data.csv"
        df = pd.DataFrame({
            'metric_code': ['INVALID'],
            'value': ['not_a_number'],
            'unit': ['invalid_unit']
        })
        df.to_csv(input_csv, index=False)

        # Run pipeline (should handle error)
        result = pipeline.run(input_path=input_csv, output_dir=tmp_path)

        assert result['status'] == 'error'
        assert 'error_stage' in result
        assert 'error_message' in result
```

**Success Criteria:**
- ✅ ReportingAgent: 85% test coverage
- ✅ Integration tests pass
- ✅ XBRL/iXBRL validation working
- ✅ End-to-end pipeline verified

**Time Estimate:** 8 hours

---

### **Day 5: Phase 6 & 7 - Scripts, Utilities, Examples & Documentation**

**Objective:** Complete remaining Phase 5-7 work

**Phase 6: Scripts & Utilities**

```python
# scripts/benchmark_performance.py

"""
Performance benchmarking script for CSRD pipeline
"""

import time
import pandas as pd
from csrd_pipeline import CSRDPipeline
from pathlib import Path
import json

def benchmark_calculator_agent():
    """Benchmark CalculatorAgent performance"""
    from agents.calculator_agent import CalculatorAgent

    calculator = CalculatorAgent()

    # Test single calculation
    input_data = {'natural_gas_m3': 45000, 'diesel_L': 2500}

    times = []
    for i in range(1000):
        start = time.time()
        result = calculator.calculate_metric('E1-1', input_data)
        times.append(time.time() - start)

    print(f"✅ CalculatorAgent Performance:")
    print(f"   - Average: {sum(times)/len(times)*1000:.2f}ms")
    print(f"   - Min: {min(times)*1000:.2f}ms")
    print(f"   - Max: {max(times)*1000:.2f}ms")
    print(f"   - Throughput: {1000/sum(times):.0f} calculations/sec")

def benchmark_intake_agent():
    """Benchmark IntakeAgent throughput"""
    from agents.intake_agent import IntakeAgent

    intake = IntakeAgent()

    # Create 10,000 record dataset
    df = pd.DataFrame({
        'metric_code': ['E1-1'] * 10000,
        'value': [9072.0] * 10000,
        'unit': ['tCO2e'] * 10000
    })

    # Save to CSV
    csv_path = Path('temp_benchmark.csv')
    df.to_csv(csv_path, index=False)

    start = time.time()
    result = intake.ingest(csv_path)
    duration = time.time() - start

    throughput = 10000 / duration

    print(f"✅ IntakeAgent Performance:")
    print(f"   - Records: 10,000")
    print(f"   - Duration: {duration:.2f}s")
    print(f"   - Throughput: {throughput:.0f} records/sec")

    csv_path.unlink()  # Clean up

def benchmark_full_pipeline():
    """Benchmark end-to-end pipeline"""
    pipeline = CSRDPipeline()

    # Create 1,000 record dataset
    df = pd.DataFrame({
        'metric_code': ['E1-1', 'E1-2'] * 500,
        'value': [9072.0, 577.5] * 500,
        'unit': ['tCO2e', 'tCO2e'] * 500
    })

    csv_path = Path('temp_pipeline_benchmark.csv')
    df.to_csv(csv_path, index=False)

    start = time.time()
    result = pipeline.run(input_path=csv_path, output_dir=Path('temp_output'))
    duration = time.time() - start

    print(f"✅ Full Pipeline Performance:")
    print(f"   - Records: 1,000")
    print(f"   - Duration: {duration:.2f}s")
    print(f"   - Target: <30 minutes for 10,000 records")
    print(f"   - Projected for 10,000: {duration*10:.2f}s ({duration*10/60:.1f} min)")

    csv_path.unlink()

if __name__ == "__main__":
    print("="*60)
    print("CSRD Pipeline Performance Benchmark")
    print("="*60)

    benchmark_calculator_agent()
    print()
    benchmark_intake_agent()
    print()
    benchmark_full_pipeline()

    print("="*60)
    print("✅ Benchmark Complete")
```

```bash
# scripts/validate_schemas.sh

#!/bin/bash
# Validate all JSON schemas

echo "Validating JSON Schemas..."

for schema in schemas/*.schema.json; do
    echo "Checking $schema..."
    python -m jsonschema --version &> /dev/null || pip install jsonschema
    python -c "import json; json.load(open('$schema'))"

    if [ $? -eq 0 ]; then
        echo "✅ $schema valid"
    else
        echo "❌ $schema invalid"
        exit 1
    fi
done

echo "✅ All schemas valid"
```

**Phase 7: Examples & Documentation**

```python
# examples/quick_start.py

"""
Quick Start Example: CSRD Reporting in 5 Minutes
"""

from csrd_pipeline import CSRDPipeline
from sdk.csrd_sdk import generate_csrd_report
import pandas as pd

def example_1_simple_api():
    """Example 1: One-function API call"""
    print("="*60)
    print("Example 1: Simple API Call")
    print("="*60)

    # Prepare your ESG data
    esg_data = pd.DataFrame({
        'metric_code': ['E1-1', 'E1-2', 'S1-1'],
        'value': [9072.0, 577.5, 325],
        'unit': ['tCO2e', 'tCO2e', 'FTE'],
        'reporting_year': [2024, 2024, 2024]
    })

    # Company profile
    company_profile = {
        'name': 'GreenTech Manufacturing GmbH',
        'lei': 'DE123456789012345678',
        'industry': 'Manufacturing',
        'revenue_million_eur': 50.0,
        'employees': 325
    }

    # Generate complete CSRD report in one call
    result = generate_csrd_report(
        esg_data=esg_data,
        company_profile=company_profile,
        output_format='xbrl',
        output_dir='output/'
    )

    print(f"✅ Report generated: {result['output_path']}")
    print(f"✅ Audit package: {result['audit_package_path']}")
    print(f"✅ Status: {result['status']}")

def example_2_pipeline_approach():
    """Example 2: Step-by-step pipeline"""
    print("="*60)
    print("Example 2: Step-by-Step Pipeline")
    print("="*60)

    pipeline = CSRDPipeline()

    # Stage 1: Ingest data
    intake_result = pipeline.intake('examples/demo_esg_data.csv')
    print(f"✅ Ingested {intake_result['records_ingested']} records")

    # Stage 2: Calculate metrics
    calc_result = pipeline.calculate()
    print(f"✅ Calculated {calc_result['metrics_calculated']} metrics")

    # Stage 3: Audit compliance
    audit_result = pipeline.audit()
    print(f"✅ Audit: {audit_result['compliance_score']:.1f}%")

    # Stage 4: Generate report
    report_result = pipeline.generate_report(format='xbrl')
    print(f"✅ Report: {report_result['output_path']}")

def example_3_custom_workflow():
    """Example 3: Custom workflow with individual agents"""
    print("="*60)
    print("Example 3: Custom Workflow")
    print("="*60)

    from agents.intake_agent import IntakeAgent
    from agents.calculator_agent import CalculatorAgent
    from agents.audit_agent import AuditAgent

    # Initialize agents
    intake = IntakeAgent()
    calculator = CalculatorAgent()
    auditor = AuditAgent()

    # Load data
    data = intake.ingest('examples/demo_esg_data.csv')

    # Calculate specific metric
    emissions = calculator.calculate_metric(
        metric_code='E1-1',
        input_data={'natural_gas_m3': 45000, 'diesel_L': 2500}
    )

    print(f"✅ Scope 1 Emissions: {emissions['value']} {emissions['unit']}")
    print(f"✅ Provenance Hash: {emissions['provenance_hash']}")

    # Audit result
    audit = auditor.execute_rule('ESRS-E1-R1', emissions)
    print(f"✅ Compliance: {audit['status']}")

if __name__ == "__main__":
    # Run all examples
    example_1_simple_api()
    print()
    example_2_pipeline_approach()
    print()
    example_3_custom_workflow()

    print("="*60)
    print("✅ All examples completed successfully!")
    print("="*60)
```

**Update README.md:**

```markdown
# GL-CSRD-APP: Enterprise CSRD/ESRS Reporting Platform

## Quick Start (5 Minutes)

### Installation

```bash
# Clone repository
git clone https://github.com/greenlang/GL-CSRD-APP
cd GL-CSRD-APP/CSRD-Reporting-Platform

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your configuration
```

### One-Line Usage

```python
from sdk.csrd_sdk import generate_csrd_report
import pandas as pd

# Your ESG data
esg_data = pd.DataFrame({
    'metric_code': ['E1-1', 'E1-2'],
    'value': [9072.0, 577.5],
    'unit': ['tCO2e', 'tCO2e']
})

# Generate complete CSRD report
result = generate_csrd_report(
    esg_data=esg_data,
    company_profile={'name': 'Your Company'},
    output_format='xbrl'
)

print(f"Report: {result['output_path']}")
```

### Features

✅ **Zero-Hallucination Guarantee** - 100% deterministic calculations
✅ **1,082 ESRS Data Points** - Complete coverage
✅ **6-Agent Pipeline** - Intake → Calculate → Audit → Report
✅ **XBRL/iXBRL/ESEF** - Digital reporting formats
✅ **Multi-Framework** - TCFD/GRI/SASB → ESRS conversion
✅ **Audit-Ready** - Complete provenance tracking (SHA-256)

### Performance

- **<30 minutes** for 10,000 data points
- **1,000+ records/sec** ingestion
- **500+ metrics/sec** calculation
- **<3 minutes** compliance validation

### Documentation

- [Complete Development Guide](docs/COMPLETE_DEVELOPMENT_GUIDE.md)
- [Development Roadmap](docs/DEVELOPMENT_ROADMAP_DETAILED.md)
- [Agent Orchestration](docs/AGENT_ORCHESTRATION_GUIDE.md)
- [Quick Start Examples](examples/quick_start.py)

### Support

- Issues: [GitHub Issues](https://github.com/greenlang/GL-CSRD-APP/issues)
- Documentation: [Full Docs](docs/)
- Contact: support@greenlang.com
```

**Success Criteria:**
- ✅ Benchmark script created and tested
- ✅ Schema validation script working
- ✅ Quick start examples functional
- ✅ README.md updated
- ✅ Documentation complete

**Time Estimate:** 6-8 hours

---

## 3.3 Week 2: Production Readiness & Agent Integration (Days 6-10)

### **Day 6-7: Phase 8 - Final Integration & Production Preparation**

**Objective:** Prepare application for production deployment

**Tasks:**
1. End-to-end testing with real data
2. Performance optimization
3. Error handling improvements
4. Logging and monitoring setup
5. Security audit

**End-to-End Testing:**

```python
# tests/e2e/test_real_data.py

import pytest
from csrd_pipeline import CSRDPipeline
from pathlib import Path

class TestRealDataScenarios:
    """Test with real-world data scenarios"""

    def test_manufacturing_company_report(self):
        """Test complete report for manufacturing company"""
        pipeline = CSRDPipeline()

        result = pipeline.run(
            input_path='tests/fixtures/manufacturing_company_2024.csv',
            company_profile='tests/fixtures/manufacturing_profile.json',
            output_dir='tests/output/manufacturing'
        )

        assert result['status'] == 'success'
        assert result['metrics_calculated'] >= 100
        assert result['compliance_score'] >= 85.0

    def test_financial_services_report(self):
        """Test financial services company report"""
        pipeline = CSRDPipeline()

        result = pipeline.run(
            input_path='tests/fixtures/financial_services_2024.csv',
            company_profile='tests/fixtures/financial_profile.json',
            output_dir='tests/output/financial'
        )

        assert result['status'] == 'success'

    def test_retail_company_report(self):
        """Test retail company report"""
        pipeline = CSRDPipeline()

        result = pipeline.run(
            input_path='tests/fixtures/retail_company_2024.csv',
            company_profile='tests/fixtures/retail_profile.json',
            output_dir='tests/output/retail'
        )

        assert result['status'] == 'success'
```

**Performance Optimization:**

```python
# Optimization checklist:

1. Database query optimization
   - Add indexes to frequently queried fields
   - Use connection pooling
   - Implement query caching

2. Calculation engine optimization
   - Vectorize calculations where possible
   - Use numpy for array operations
   - Implement result caching for repeated calculations

3. Memory optimization
   - Use generators for large datasets
   - Implement streaming for file I/O
   - Clear intermediate results

4. Parallel processing
   - Use multiprocessing for independent calculations
   - Implement batch processing
   - Parallelize agent execution where possible
```

**Logging & Monitoring:**

```python
# utils/logging_config.py

import logging
import sys
from pathlib import Path

def setup_logging(log_level='INFO', log_file=None):
    """Configure logging for CSRD pipeline"""

    # Create logger
    logger = logging.getLogger('csrd_pipeline')
    logger.setLevel(getattr(logging, log_level))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

# Monitoring utilities

class PerformanceMonitor:
    """Monitor pipeline performance metrics"""

    def __init__(self):
        self.metrics = {}

    def record_metric(self, name, value, unit=''):
        """Record a performance metric"""
        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append({
            'value': value,
            'unit': unit,
            'timestamp': time.time()
        })

    def get_summary(self):
        """Get summary of all metrics"""
        summary = {}
        for name, values in self.metrics.items():
            summary[name] = {
                'count': len(values),
                'average': sum(v['value'] for v in values) / len(values),
                'min': min(v['value'] for v in values),
                'max': max(v['value'] for v in values)
            }
        return summary
```

**Security Audit:**

```python
# security/security_checklist.py

"""
Security Audit Checklist for CSRD Pipeline

✅ Input Validation
- All user inputs validated against schemas
- SQL injection prevention (parameterized queries)
- Path traversal prevention (sanitize file paths)
- File upload size limits enforced

✅ Data Protection
- Sensitive data encrypted at rest
- Secure credential management (environment variables, not hardcoded)
- API keys rotated regularly
- Database credentials secured

✅ Output Security
- Generated files have proper permissions
- Temporary files cleaned up
- Audit logs protected from tampering

✅ Dependency Security
- All dependencies scanned for vulnerabilities
- Regular updates applied
- License compliance verified

✅ API Security
- Rate limiting implemented
- Authentication required
- HTTPS enforced
- CORS properly configured
"""

def run_security_scan():
    """Run automated security scan"""
    import subprocess

    print("Running security scan...")

    # Dependency vulnerability scan
    print("1. Scanning dependencies...")
    subprocess.run(['safety', 'check', '--json'])

    # Code security scan
    print("2. Scanning code...")
    subprocess.run(['bandit', '-r', '.', '-f', 'json'])

    # Secret detection
    print("3. Scanning for secrets...")
    subprocess.run(['detect-secrets', 'scan', '--all-files'])

    print("✅ Security scan complete")

if __name__ == "__main__":
    run_security_scan()
```

**Success Criteria:**
- ✅ E2E tests pass with real data
- ✅ Performance meets all targets
- ✅ Logging and monitoring configured
- ✅ Security audit passed
- ✅ Production deployment ready

**Time Estimate:** 12-16 hours (2 days)

---

### **Day 8-10: Phase 9 - GreenLang Agent Integration (14 Agents)**

**Objective:** Integrate 14 GreenLang platform agents with CSRD pipeline

**GreenLang Agent Ecosystem:**

1. **GL-CodeSentinel** - Code quality and linting
2. **GL-SecScan** - Security scanning
3. **GL-SpecGuardian** - Specification validation
4. **GL-SupplyChainSentinel** - Supply chain security
5. **GL-PolicyLinter** - OPA policy linting
6. **GL-PackQC** - Package quality control
7. **GL-HubRegistrar** - Package publishing
8. **GL-ExitBarAuditor** - Release gate validation
9. **GL-DeterminismAuditor** - Reproducibility verification
10. **GL-DataFlowGuardian** - Data lineage tracking
11. **GL-ConnectorValidator** - Connector validation
12. **GL-TaskChecker** - Task completion verification
13. **GL-ProductDevelopmentTracker** - Development tracking
14. **GL-ProjectStatusReporter** - Status reporting

**Integration Configuration:**

```yaml
# config/greenlang_agents_config.yaml

greenlang_agents:
  enabled: true

  # Development Quality Workflow
  development_quality:
    - gl_codesentinel:
        enabled: true
        config:
          lint_rules: ".pylintrc"
          type_check: true
          max_complexity: 10

    - gl_secscan:
        enabled: true
        config:
          scan_dependencies: true
          scan_secrets: true
          fail_on_high: true

    - gl_spec_guardian:
        enabled: true
        config:
          validate_pack_yaml: true
          validate_gl_yaml: true
          validate_policy_schemas: true

  # Data Pipeline Workflow
  data_pipeline:
    - gl_dataflow_guardian:
        enabled: true
        config:
          trace_lineage: true
          validate_transformations: true
          check_data_quality: true

    - gl_determinism_auditor:
        enabled: true
        config:
          verify_reproducibility: true
          hash_algorithm: "sha256"
          tolerance: 0.0  # Zero tolerance for differences

  # Release Readiness Workflow
  release_readiness:
    - gl_packqc:
        enabled: true
        config:
          validate_dependencies: true
          check_version_compatibility: true
          verify_metadata: true

    - gl_exitbar_auditor:
        enabled: true
        config:
          quality_gate_threshold: 90
          security_gate_threshold: 95
          performance_gate_threshold: 85

    - gl_hub_registrar:
        enabled: true
        config:
          registry_url: "https://hub.greenlang.com"
          auto_publish: false

  # Project Management Workflow
  project_management:
    - gl_task_checker:
        enabled: true
        config:
          verify_requirements: true
          check_test_coverage: true

    - gl_product_development_tracker:
        enabled: true
        config:
          track_features: true
          track_bugs: true
          track_performance: true

    - gl_project_status_reporter:
        enabled: true
        config:
          generate_daily_reports: true
          notify_stakeholders: true
```

**CI/CD Integration:**

```yaml
# .github/workflows/csrd_quality_gates.yml

name: CSRD Quality Gates

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  code-quality:
    name: Code Quality Gate
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: GL-CodeSentinel - Code Quality Check
        run: |
          python -m greenlang.agents.gl_codesentinel \
            --path . \
            --output quality_report.json \
            --fail-on-error

      - name: GL-SecScan - Security Scan
        run: |
          python -m greenlang.agents.gl_secscan \
            --path . \
            --output security_report.json \
            --fail-on-high

      - name: Upload Reports
        uses: actions/upload-artifact@v3
        with:
          name: quality-reports
          path: |
            quality_report.json
            security_report.json

  testing:
    name: Testing Gate
    runs-on: ubuntu-latest
    needs: code-quality

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run Unit Tests
        run: |
          pytest tests/unit/ \
            --cov=agents \
            --cov-report=xml \
            --cov-report=html \
            --cov-fail-under=90

      - name: Run Integration Tests
        run: |
          pytest tests/integration/ -v

      - name: GL-TaskChecker - Verify Test Coverage
        run: |
          python -m greenlang.agents.gl_task_checker \
            --task "Verify test coverage >=90%" \
            --evidence coverage.xml

  data-quality:
    name: Data Quality Gate
    runs-on: ubuntu-latest
    needs: testing

    steps:
      - uses: actions/checkout@v3

      - name: GL-DataFlowGuardian - Validate Data Pipelines
        run: |
          python -m greenlang.agents.gl_dataflow_guardian \
            --pipeline-config config/csrd_config.yaml \
            --validate-lineage \
            --output dataflow_report.json

      - name: GL-DeterminismAuditor - Verify Reproducibility
        run: |
          python -m greenlang.agents.gl_determinism_auditor \
            --run-twice \
            --compare-hashes \
            --tolerance 0.0

  release-gate:
    name: Release Readiness Gate
    runs-on: ubuntu-latest
    needs: [code-quality, testing, data-quality]
    if: github.ref == 'refs/heads/main'

    steps:
      - uses: actions/checkout@v3

      - name: GL-PackQC - Package Quality Control
        run: |
          python -m greenlang.agents.gl_packqc \
            --validate-all \
            --output packqc_report.json

      - name: GL-ExitBarAuditor - Release Gate Validation
        run: |
          python -m greenlang.agents.gl_exitbar_auditor \
            --quality-threshold 90 \
            --security-threshold 95 \
            --performance-threshold 85 \
            --output exitbar_report.json

      - name: Check Exit Bar Status
        run: |
          if [ $(jq '.status' exitbar_report.json) == "PASS" ]; then
            echo "✅ Release gate PASSED"
          else
            echo "❌ Release gate FAILED"
            jq '.failures' exitbar_report.json
            exit 1
          fi

      - name: GL-ProjectStatusReporter - Generate Release Report
        run: |
          python -m greenlang.agents.gl_project_status_reporter \
            --generate-report \
            --include-all-gates \
            --output release_report.md

      - name: Upload Release Report
        uses: actions/upload-artifact@v3
        with:
          name: release-report
          path: release_report.md
```

**Agent Orchestration Implementation:**

```python
# utils/agent_orchestrator.py

from typing import List, Dict, Any
import asyncio
from pathlib import Path
import yaml

class GreenLangAgentOrchestrator:
    """Orchestrate GreenLang agents for CSRD pipeline"""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.agents = self._initialize_agents()

    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all enabled agents"""
        agents = {}

        for workflow_name, workflow_agents in self.config['greenlang_agents'].items():
            if workflow_name == 'enabled':
                continue

            for agent_config in workflow_agents:
                for agent_name, agent_settings in agent_config.items():
                    if agent_settings.get('enabled'):
                        agents[agent_name] = self._load_agent(agent_name, agent_settings)

        return agents

    def _load_agent(self, agent_name: str, settings: Dict[str, Any]):
        """Dynamically load agent module"""
        import importlib

        module_name = f"greenlang.agents.{agent_name}"
        agent_module = importlib.import_module(module_name)
        agent_class = getattr(agent_module, self._to_class_name(agent_name))

        return agent_class(settings.get('config', {}))

    def _to_class_name(self, agent_name: str) -> str:
        """Convert agent name to class name"""
        # gl_codesentinel → GLCodeSentinel
        parts = agent_name.split('_')
        return ''.join(part.capitalize() for part in parts)

    async def run_workflow(self, workflow_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow (sequence of agents)"""
        workflow_config = self.config['greenlang_agents'].get(workflow_name)

        if not workflow_config:
            raise ValueError(f"Workflow {workflow_name} not found")

        results = {}

        for agent_config in workflow_config:
            for agent_name, agent_settings in agent_config.items():
                if not agent_settings.get('enabled'):
                    continue

                print(f"Running {agent_name}...")

                agent = self.agents[agent_name]
                result = await agent.run(context)

                results[agent_name] = result

                # Check if agent failed and should block pipeline
                if result.get('status') == 'FAIL' and agent_settings.get('fail_on_error'):
                    raise Exception(f"Agent {agent_name} failed: {result.get('message')}")

        return results

    async def run_all_workflows(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run all workflows in parallel"""
        workflows = ['development_quality', 'data_pipeline', 'release_readiness']

        tasks = [
            self.run_workflow(workflow, context)
            for workflow in workflows
        ]

        results = await asyncio.gather(*tasks)

        return dict(zip(workflows, results))

# Usage example

async def main():
    orchestrator = GreenLangAgentOrchestrator('config/greenlang_agents_config.yaml')

    context = {
        'project_path': '.',
        'branch': 'main',
        'commit_sha': 'abc123'
    }

    # Run development quality workflow
    dev_results = await orchestrator.run_workflow('development_quality', context)

    print("Development Quality Results:")
    for agent, result in dev_results.items():
        print(f"  {agent}: {result['status']}")

    # Run all workflows
    all_results = await orchestrator.run_all_workflows(context)

    print("\nAll Workflows Complete:")
    for workflow, agents in all_results.items():
        print(f"\n{workflow}:")
        for agent, result in agents.items():
            print(f"  {agent}: {result['status']}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Success Criteria:**
- ✅ All 14 GreenLang agents integrated
- ✅ 4 orchestration workflows configured
- ✅ CI/CD pipelines implemented
- ✅ Agent orchestrator tested
- ✅ Documentation updated

**Time Estimate:** 18-24 hours (3 days)

---

*Document continues in COMPLETE_DEVELOPMENT_GUIDE_PART3.md...*
