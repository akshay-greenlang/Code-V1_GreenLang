# TECHNICAL REQUIREMENTS DOCUMENT
# GL-SB253-APP: US State Climate Disclosure Platform

**Version:** 1.0
**Date:** November 10, 2025
**Status:** TIER 1 - EXTREME URGENCY
**Compliance Deadline:** June 30, 2026

---

## 1. AGENT PIPELINE SPECIFICATIONS

### 1.1 DataCollectionAgent

**Purpose:** Automated data collection from multiple sources for Scope 1, 2, and 3 emissions.

**Capabilities:**
```python
class DataCollectionAgent:
    """
    Automated multi-source data collection agent
    """

    data_sources = {
        "erp_systems": ["SAP", "Oracle", "Workday"],  # Reuse GL-VCCI
        "utility_apis": ["electricity", "natural_gas", "water"],
        "fleet_systems": ["Geotab", "Samsara", "Verizon Connect"],
        "travel_systems": ["Concur", "Expensify", "TripActions"],
        "manual_upload": ["Excel", "CSV", "JSON"]
    }

    async def collect_scope_1_data(self):
        """Stationary and mobile combustion"""
        return {
            "natural_gas": self.get_utility_data("gas"),
            "fleet_fuel": self.get_fleet_data(),
            "refrigerants": self.get_facility_data()
        }

    async def collect_scope_2_data(self):
        """Purchased electricity, steam, heating, cooling"""
        return {
            "electricity": self.get_utility_data("electricity"),
            "steam": self.get_facility_data("steam"),
            "renewable_energy": self.get_renewable_certificates()
        }

    async def collect_scope_3_data(self):
        """Leverage GL-VCCI-APP for all 15 categories"""
        from gl_vcci.agents import Scope3CalculatorAgent
        return Scope3CalculatorAgent.collect_all_categories()
```

**Integration Points:**
- ERP Connectors (from GL-VCCI)
- Utility API Gateway
- Fleet Management APIs
- Travel System APIs

**Data Quality Requirements:**
- Completeness: > 90% for Scope 1&2
- Accuracy: ± 5% variance
- Timeliness: < 30 days old

### 1.2 CalculationAgent

**Purpose:** GHG Protocol compliant calculations with zero hallucination for actual data.

**Architecture:**
```python
class CalculationAgent:
    """
    Deterministic calculation engine - NO HALLUCINATION
    """

    def __init__(self):
        # Import GL-VCCI calculation engines
        from gl_vcci.services.methodologies import (
            Scope1Calculator,
            Scope2Calculator,
            Scope3Calculator
        )
        self.scope1_engine = Scope1Calculator()
        self.scope2_engine = Scope2Calculator()
        self.scope3_engine = Scope3Calculator()

    def calculate_emissions(self, activity_data):
        """
        Calculate emissions using GHG Protocol methodology
        """
        results = {
            "scope_1": self.scope1_engine.calculate(activity_data.scope1),
            "scope_2": {
                "location_based": self.scope2_engine.location_based(activity_data.scope2),
                "market_based": self.scope2_engine.market_based(activity_data.scope2)
            },
            "scope_3": self.scope3_engine.calculate_all_categories(activity_data.scope3)
        }

        # Add provenance
        results["provenance"] = self.generate_provenance(activity_data, results)

        return results

    def generate_provenance(self, inputs, outputs):
        """
        Create SHA-256 hash chain for audit trail
        """
        import hashlib
        import json

        provenance = {
            "timestamp": datetime.utcnow().isoformat(),
            "input_hash": hashlib.sha256(json.dumps(inputs).encode()).hexdigest(),
            "output_hash": hashlib.sha256(json.dumps(outputs).encode()).hexdigest(),
            "methodology": "GHG Protocol Corporate Standard v1.0",
            "factors_version": "EPA 2025.1, DEFRA 2025"
        }

        return provenance
```

**Calculation Requirements:**
- Zero hallucination for actual data
- AI estimation only when data unavailable
- Complete audit trail for every calculation
- Reproducible results (deterministic)

### 1.3 AssuranceReadyAgent

**Purpose:** Generate complete audit packages for third-party assurance.

**Specifications:**
```python
class AssuranceReadyAgent:
    """
    Generate third-party assurance packages
    """

    def generate_assurance_package(self, company_id, reporting_year):
        """
        Create complete audit package for Big 4 firms
        """

        package = {
            "executive_summary": self.create_summary(),
            "organizational_boundaries": self.document_boundaries(),
            "operational_boundaries": self.document_operations(),
            "base_year": self.document_base_year(),
            "emissions_data": {
                "scope_1": self.compile_scope1_evidence(),
                "scope_2": self.compile_scope2_evidence(),
                "scope_3": self.compile_scope3_evidence()
            },
            "methodology": self.document_methodology(),
            "emission_factors": self.list_all_factors(),
            "data_quality": self.assess_data_quality(),
            "uncertainty": self.calculate_uncertainty(),
            "provenance_chain": self.export_full_provenance(),
            "supporting_documents": self.gather_evidence()
        }

        # Sign package
        package["digital_signature"] = self.sign_package(package)

        return package

    def compile_scope1_evidence(self):
        """
        Complete evidence for Scope 1 emissions
        """
        return {
            "activity_data": self.get_source_documents(),
            "calculation_sheets": self.export_calculations(),
            "emission_factors": self.list_factors_used(),
            "quality_checks": self.validation_results(),
            "data_gaps": self.document_estimations()
        }

    def assess_data_quality(self):
        """
        Data Quality Indicators (DQI) per GHG Protocol
        """
        return {
            "temporal_correlation": 0.95,  # How recent is data
            "geographical_correlation": 0.98,  # Location specificity
            "technological_correlation": 0.92,  # Technology match
            "completeness": 0.94,  # Data coverage
            "reliability": 0.96  # Source reliability
        }
```

**Assurance Requirements:**
- Limited assurance (2026-2029)
- Reasonable assurance (2030+)
- Big 4 firm compatible formats
- Complete evidence chain

### 1.4 MultiStateFilingAgent

**Purpose:** Handle state-specific compliance and filing requirements.

**Architecture:**
```python
class MultiStateFilingAgent:
    """
    Multi-state compliance and filing engine
    """

    STATE_PORTALS = {
        "CA": {
            "portal": "CARB",
            "api_endpoint": "https://ww2.arb.ca.gov/ghg-api",
            "submission_format": "JSON",
            "deadline": "June 30"
        },
        "CO": {
            "portal": "Colorado APCD",
            "api_endpoint": "TBD",
            "submission_format": "XML",
            "deadline": "June 30"
        },
        "WA": {
            "portal": "Dept of Ecology",
            "api_endpoint": "TBD",
            "submission_format": "JSON",
            "deadline": "October 1"
        }
    }

    async def file_disclosure(self, state, disclosure_data):
        """
        File disclosure with state portal
        """
        portal = self.STATE_PORTALS[state]

        # Format data per state requirements
        formatted_data = self.format_for_state(state, disclosure_data)

        # Validate against state rules
        validation = self.validate_state_requirements(state, formatted_data)
        if not validation.is_valid:
            raise ValidationError(validation.errors)

        # Submit to state portal
        if state == "CA":
            return await self.submit_to_carb(formatted_data)
        elif state == "CO":
            return await self.submit_to_colorado(formatted_data)
        elif state == "WA":
            return await self.submit_to_washington(formatted_data)

    async def submit_to_carb(self, data):
        """
        California Air Resources Board submission
        """
        payload = {
            "disclosure_year": data.year,
            "entity": {
                "name": data.company_name,
                "ein": data.ein,
                "revenue": data.revenue
            },
            "emissions": {
                "scope_1": data.scope_1,
                "scope_2_location": data.scope_2_location,
                "scope_2_market": data.scope_2_market,
                "scope_3": data.scope_3_by_category if data.year >= 2027 else None
            },
            "assurance": {
                "provider": data.assurance_provider,
                "level": data.assurance_level,
                "report_url": data.assurance_report
            }
        }

        response = await self.post_to_portal("CA", payload)
        return response.filing_id
```

**State Requirements:**
- California SB 253 compliance
- Colorado HB 25-1119 readiness
- Washington SB 6092 support
- Extensible for new states

### 1.5 ThirdPartyAssuranceAgent

**Purpose:** Interface with Big 4 audit firms for verification.

**Specifications:**
```python
class ThirdPartyAssuranceAgent:
    """
    Third-party assurance integration
    """

    ASSURANCE_PROVIDERS = {
        "PwC": {
            "portal": "PwC Halo",
            "format": "XBRL",
            "api_available": True
        },
        "EY": {
            "portal": "EY Helix",
            "format": "JSON",
            "api_available": True
        },
        "Deloitte": {
            "portal": "Deloitte Omnia",
            "format": "XML",
            "api_available": False
        },
        "KPMG": {
            "portal": "KPMG Clara",
            "format": "JSON",
            "api_available": True
        }
    }

    def prepare_assurance_package(self, provider):
        """
        Prepare provider-specific assurance package
        """
        package = {
            "inventory": self.export_ghg_inventory(),
            "methodology": self.export_methodology(),
            "evidence": self.compile_evidence(),
            "calculations": self.export_calculations(),
            "management_assertion": self.generate_assertion(),
            "representation_letter": self.draft_representation_letter()
        }

        # Format per provider requirements
        if provider in self.ASSURANCE_PROVIDERS:
            format_type = self.ASSURANCE_PROVIDERS[provider]["format"]
            return self.format_package(package, format_type)

        return package

    def track_assurance_process(self):
        """
        Track assurance engagement progress
        """
        return {
            "planning": {
                "status": "complete",
                "risk_assessment": "documented",
                "materiality": "calculated"
            },
            "evidence_gathering": {
                "status": "in_progress",
                "samples_requested": 150,
                "samples_provided": 145,
                "outstanding_items": 5
            },
            "testing": {
                "status": "pending",
                "procedures": "defined",
                "timeline": "2 weeks"
            },
            "reporting": {
                "status": "not_started",
                "draft_date": "TBD",
                "final_date": "TBD"
            }
        }
```

---

## 2. CARB PORTAL INTEGRATION

### 2.1 API Specifications (Expected)

```yaml
CARB API Specification:
  base_url: https://ww2.arb.ca.gov/ghg-disclosure-api
  version: v1
  authentication: OAuth2

  endpoints:
    - /auth/token
      method: POST
      description: Obtain access token

    - /organizations
      method: POST
      description: Register organization

    - /disclosures
      method: POST
      description: Submit annual disclosure

    - /disclosures/{id}
      method: GET
      description: Retrieve disclosure status

    - /disclosures/{id}/amend
      method: PUT
      description: Amend submitted disclosure

    - /assurance
      method: POST
      description: Submit assurance report

  rate_limits:
    - requests_per_minute: 60
    - requests_per_day: 1000
    - payload_size_mb: 50
```

### 2.2 Fallback Strategy

If CARB API is not available by launch:

1. **Manual Upload Portal:**
   - Generate CARB-compliant PDF reports
   - Create structured data files (JSON/XML)
   - Provide submission checklist

2. **Email Submission:**
   - Automated email generation
   - Attachment validation
   - Delivery confirmation tracking

3. **Partner Integration:**
   - Work with CARB-approved third parties
   - Use existing environmental reporting platforms
   - Leverage audit firm submission channels

---

## 3. DATA SCHEMAS

### 3.1 Core Emissions Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "SB253 Emissions Disclosure",
  "type": "object",
  "required": ["organization", "reporting_year", "emissions", "assurance"],
  "properties": {
    "organization": {
      "type": "object",
      "required": ["name", "ein", "revenue", "ca_operations"],
      "properties": {
        "name": {"type": "string"},
        "ein": {"type": "string", "pattern": "^[0-9]{2}-[0-9]{7}$"},
        "revenue": {"type": "number", "minimum": 1000000000},
        "ca_operations": {"type": "boolean"},
        "headquarters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"},
            "state": {"type": "string", "maxLength": 2},
            "country": {"type": "string", "maxLength": 2}
          }
        }
      }
    },
    "reporting_year": {
      "type": "integer",
      "minimum": 2025
    },
    "emissions": {
      "type": "object",
      "required": ["scope_1", "scope_2"],
      "properties": {
        "scope_1": {
          "type": "object",
          "required": ["total", "by_source"],
          "properties": {
            "total": {"type": "number", "minimum": 0},
            "by_source": {
              "type": "object",
              "properties": {
                "stationary_combustion": {"type": "number"},
                "mobile_combustion": {"type": "number"},
                "process_emissions": {"type": "number"},
                "fugitive_emissions": {"type": "number"}
              }
            }
          }
        },
        "scope_2": {
          "type": "object",
          "required": ["location_based", "market_based"],
          "properties": {
            "location_based": {"type": "number", "minimum": 0},
            "market_based": {"type": "number", "minimum": 0}
          }
        },
        "scope_3": {
          "type": "object",
          "properties": {
            "total": {"type": "number", "minimum": 0},
            "by_category": {
              "type": "object",
              "properties": {
                "category_1": {"type": "number"},
                "category_2": {"type": "number"},
                "category_3": {"type": "number"},
                "category_4": {"type": "number"},
                "category_5": {"type": "number"},
                "category_6": {"type": "number"},
                "category_7": {"type": "number"},
                "category_8": {"type": "number"},
                "category_9": {"type": "number"},
                "category_10": {"type": "number"},
                "category_11": {"type": "number"},
                "category_12": {"type": "number"},
                "category_13": {"type": "number"},
                "category_14": {"type": "number"},
                "category_15": {"type": "number"}
              }
            }
          }
        }
      }
    },
    "assurance": {
      "type": "object",
      "required": ["provider", "level", "statement"],
      "properties": {
        "provider": {"type": "string"},
        "level": {
          "type": "string",
          "enum": ["limited", "reasonable"]
        },
        "statement": {"type": "string"},
        "report_date": {
          "type": "string",
          "format": "date"
        }
      }
    }
  }
}
```

---

## 4. PERFORMANCE REQUIREMENTS

### 4.1 System Performance

| Metric | Requirement | Measurement Method |
|--------|------------|-------------------|
| **Data Processing** | < 5 min for 10,000 records | Load testing |
| **Calculation Time** | < 10 sec per company | Unit tests |
| **Report Generation** | < 30 sec for full report | End-to-end test |
| **API Response** | < 200ms for 95th percentile | APM monitoring |
| **Concurrent Users** | Support 100 simultaneous | Load testing |
| **Data Volume** | Handle 1TB emissions data | Database capacity |

### 4.2 Scalability Requirements

```yaml
Scalability Targets:
  Year 1 (2026):
    companies: 100
    data_points: 1M
    calculations_per_day: 10K

  Year 2 (2027):
    companies: 1000
    data_points: 20M
    calculations_per_day: 100K

  Year 3 (2028):
    companies: 5000
    data_points: 200M
    calculations_per_day: 1M
```

---

## 5. SECURITY REQUIREMENTS

### 5.1 Data Security

**Encryption:**
- At rest: AES-256
- In transit: TLS 1.3
- Database: Transparent Data Encryption (TDE)

**Access Control:**
- Multi-factor authentication (MFA)
- Role-based access control (RBAC)
- API key management
- OAuth2 for third-party integrations

**Audit Logging:**
- All data access logged
- Calculation audit trail
- User activity monitoring
- Compliance reporting

### 5.2 Compliance

**Standards:**
- SOC 2 Type II
- ISO 27001
- GDPR (for EU companies)
- CCPA (California privacy)

---

## 6. INTEGRATION REQUIREMENTS

### 6.1 ERP Systems (Leverage GL-VCCI)

```python
# Reuse existing connectors from GL-VCCI
from gl_vcci.connectors import (
    SAPConnector,      # SAP S/4HANA
    OracleConnector,   # Oracle ERP Cloud
    WorkdayConnector   # Workday Financial Management
)

class ERPIntegration:
    """
    Leverage GL-VCCI ERP connectors
    """
    def __init__(self):
        self.sap = SAPConnector()
        self.oracle = OracleConnector()
        self.workday = WorkdayConnector()

    async def extract_financial_data(self, erp_type, credentials):
        """
        Extract spend data for Scope 3 calculations
        """
        if erp_type == "SAP":
            return await self.sap.get_purchase_orders(credentials)
        elif erp_type == "Oracle":
            return await self.oracle.get_supplier_invoices(credentials)
        elif erp_type == "Workday":
            return await self.workday.get_expense_reports(credentials)
```

### 6.2 Third-Party APIs

**Required Integrations:**
- Utility providers (electricity, gas)
- Fleet management systems
- Travel booking systems
- Waste management systems
- Banking APIs (for financed emissions)

---

## 7. TESTING REQUIREMENTS

### 7.1 Test Coverage

| Test Type | Coverage Target | Focus Areas |
|-----------|----------------|-------------|
| **Unit Tests** | > 90% | Calculation accuracy |
| **Integration Tests** | > 80% | API endpoints |
| **E2E Tests** | > 70% | Complete workflows |
| **Performance Tests** | 100% critical paths | Data processing |
| **Security Tests** | 100% | OWASP Top 10 |

### 7.2 Compliance Testing

```python
class ComplianceTestSuite:
    """
    SB 253 compliance validation
    """

    def test_scope_1_2_calculations(self):
        """Verify GHG Protocol compliance"""
        assert calculations_follow_ghg_protocol()

    def test_assurance_package_completeness(self):
        """Ensure audit readiness"""
        assert all_required_evidence_included()

    def test_carb_submission_format(self):
        """Validate CARB requirements"""
        assert submission_meets_carb_specifications()

    def test_multi_state_requirements(self):
        """Verify state-specific rules"""
        for state in ["CA", "CO", "WA"]:
            assert state_requirements_met(state)
```

---

## 8. DEPLOYMENT REQUIREMENTS

### 8.1 Infrastructure

```yaml
Production Environment:
  compute:
    - kubernetes_cluster: 3 nodes minimum
    - cpu: 16 cores per node
    - memory: 64GB per node

  database:
    - postgresql: v14+
    - storage: 1TB SSD
    - replication: Multi-AZ

  cache:
    - redis: v7+
    - memory: 32GB

  search:
    - weaviate: v1.20+
    - storage: 500GB

  monitoring:
    - prometheus: Metrics collection
    - grafana: Dashboards
    - sentry: Error tracking
```

### 8.2 CI/CD Pipeline

```yaml
GitHub Actions Workflow:
  - name: Test
    steps:
      - pytest (unit tests)
      - integration tests
      - security scan

  - name: Build
    steps:
      - docker build
      - push to registry

  - name: Deploy
    steps:
      - staging deployment
      - smoke tests
      - production deployment
      - health checks
```

---

## 9. DOCUMENTATION REQUIREMENTS

### 9.1 Technical Documentation

- [ ] API documentation (OpenAPI/Swagger)
- [ ] Agent architecture diagrams
- [ ] Database schema documentation
- [ ] Integration guides
- [ ] Deployment runbooks

### 9.2 User Documentation

- [ ] SB 253 compliance guide
- [ ] Platform user manual
- [ ] Data collection templates
- [ ] Assurance preparation checklist
- [ ] Video tutorials

### 9.3 Compliance Documentation

- [ ] GHG Protocol methodology
- [ ] Emission factor sources
- [ ] Calculation methodology
- [ ] Data quality procedures
- [ ] Audit trail documentation

---

## APPROVAL

| Role | Name | Approval Date |
|------|------|--------------|
| Technical Lead | | |
| Product Manager | | |
| Compliance Officer | | |
| Security Officer | | |

---

**Document Version:** 1.0
**Last Updated:** November 10, 2025
**Next Review:** November 17, 2025