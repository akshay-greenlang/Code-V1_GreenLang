# Climate Science & Policy Team Charter

**Version:** 1.0
**Date:** 2025-12-03
**Team:** Climate Science & Policy
**Tech Lead:** TBD
**Headcount:** 3-4 specialists

---

## Team Mission

Ensure all generated agents are scientifically accurate, regulatory compliant, and audit-ready by providing domain validation, certification frameworks, and golden test suites rooted in authoritative climate science and EU policy.

**Core Principle:** Zero tolerance for regulatory non-compliance or scientific inaccuracy.

---

## Team Mandate

The Climate Science & Policy Team owns the domain expertise layer:

1. **Validation Hooks:** Python functions that validate agent outputs against regulations
2. **Certification Framework:** Process for certifying agents as regulation-compliant
3. **Golden Test Suites:** Domain-specific test cases with known correct answers
4. **Regulatory Intelligence:** Continuous monitoring of regulation updates

**Non-Goals:**
- Building agents (AI/Agent Team owns this)
- Code generation infrastructure (ML Platform Team owns this)
- Production deployment (DevOps Team owns this)

---

## Team Composition

### Roles & Responsibilities

**Tech Lead (1):**
- Climate Science Lead with software background
- Validation architecture and framework design
- Cross-team coordination (AI/Agent, ML Platform)
- Regulatory compliance strategy

**Climate Scientists (2):**
- Carbon accounting methodologies (CBAM, GHG Protocol)
- Life cycle assessment (LCA) expertise
- Emission factor validation and updates
- Scientific peer review

**Policy Analysts (1-2):**
- EU regulatory expertise (CBAM, EUDR, CSRD, EMAS)
- Policy change monitoring and impact analysis
- Compliance interpretation and guidance
- Certification criteria development

---

## Core Responsibilities

### 1. Validation Hooks (Regulatory Validation Framework)

**Deliverables:**

| Component | Description | Phase |
|-----------|-------------|-------|
| **CBAM Validation Hooks** | Validate CBAM calculations, CN codes, emission factors | Phase 1 |
| **EUDR Validation Hooks** | Validate geolocation, due diligence, risk assessments | Phase 2 |
| **CSRD Validation Hooks** | Validate double materiality, ESG metrics, disclosures | Phase 2 |
| **EMAS Validation Hooks** | Validate environmental statements, audits | Phase 3 |
| **Validation SDK** | Reusable library for all validation hooks | Phase 1 |

**Technical Specifications:**

**Validation Hook Architecture:**
```python
# greenlang_validation/__init__.py

from greenlang_validation.cbam import CBAMValidator
from greenlang_validation.eudr import EUDRValidator
from greenlang_validation.csrd import CSRDValidator
from greenlang_validation.hooks import ValidationHook, ValidationResult

__all__ = [
    "CBAMValidator",
    "EUDRValidator",
    "CSRDValidator",
    "ValidationHook",
    "ValidationResult",
]
```

**CBAM Validation Hook Example:**
```python
class CBAMValidator(ValidationHook):
    """
    Validate CBAM calculations against Regulation 2023/956.

    Validation checks:
    1. CN code is valid (8-digit Combined Nomenclature)
    2. Origin country is valid (ISO 3166-1 alpha-2)
    3. Emission factor is from authoritative source (IEA, IPCC)
    4. Calculation methodology matches CBAM Annex IV
    5. Embedded emissions are within expected range
    """

    def __init__(self, config: dict):
        self.cn_codes = self.load_cn_codes()  # Official EU CN code list
        self.emission_factors = self.load_emission_factors()  # IEA database
        self.country_codes = self.load_country_codes()  # ISO 3166-1

    def validate(self, agent_output: dict) -> ValidationResult:
        """
        Validate CBAM calculation.

        Args:
            agent_output: Output from CBAM agent (embedded emissions)

        Returns:
            ValidationResult with pass/fail and details
        """
        errors = []
        warnings = []

        # Check 1: CN code
        if not self.validate_cn_code(agent_output["cn_code"]):
            errors.append(f"Invalid CN code: {agent_output['cn_code']}")

        # Check 2: Origin country
        if agent_output["origin_country"] not in self.country_codes:
            errors.append(f"Invalid origin country: {agent_output['origin_country']}")

        # Check 3: Emission factor provenance
        ef_result = self.validate_emission_factor(
            product=agent_output["cn_code"],
            country=agent_output["origin_country"],
            emission_factor=agent_output["emission_factor"]
        )
        if not ef_result.is_valid:
            errors.append(f"Emission factor not from authoritative source")
            warnings.append(f"Expected: {ef_result.expected}, Got: {agent_output['emission_factor']}")

        # Check 4: Calculation methodology
        calc_result = self.validate_calculation(
            weight_kg=agent_output["weight_kg"],
            emission_factor=agent_output["emission_factor"],
            embedded_emissions=agent_output["embedded_emissions_tco2e"]
        )
        if not calc_result.is_valid:
            errors.append(f"Calculation error: {calc_result.message}")

        # Check 5: Range check
        if not self.validate_range(
            product=agent_output["cn_code"],
            embedded_emissions=agent_output["embedded_emissions_tco2e"]
        ):
            warnings.append(f"Emissions outside expected range (manual review recommended)")

        return ValidationResult(
            is_valid=(len(errors) == 0),
            errors=errors,
            warnings=warnings,
            score=self.calculate_score(errors, warnings),
            metadata={
                "validator": "CBAMValidator",
                "regulation": "CBAM Regulation 2023/956",
                "validated_at": datetime.utcnow().isoformat()
            }
        )

    def validate_emission_factor(
        self, product: str, country: str, emission_factor: float
    ) -> EmissionFactorValidation:
        """
        Validate emission factor against authoritative database.

        Authoritative sources (in priority order):
        1. IEA (International Energy Agency)
        2. IPCC (Intergovernmental Panel on Climate Change)
        3. World Steel Association (WSA)
        4. International Aluminium Institute (IAI)
        5. EU default values (CBAM Annex IV)
        """
        # Lookup expected emission factor
        expected_ef = self.emission_factors.get(
            product_category=self.get_product_category(product),
            country=country
        )

        # Allow 5% tolerance for rounding
        tolerance = 0.05
        is_valid = abs(emission_factor - expected_ef.value) / expected_ef.value < tolerance

        return EmissionFactorValidation(
            is_valid=is_valid,
            expected=expected_ef.value,
            actual=emission_factor,
            source=expected_ef.source,
            reference=expected_ef.reference_url
        )
```

**Validation Hook API Contract:**
```python
# All validation hooks must implement this interface
class ValidationHook(ABC):
    """Base class for all validation hooks."""

    @abstractmethod
    def validate(self, agent_output: dict) -> ValidationResult:
        """
        Validate agent output.

        Args:
            agent_output: Dictionary with agent-generated results

        Returns:
            ValidationResult with pass/fail and detailed feedback
        """
        pass

    def calculate_score(self, errors: list, warnings: list) -> float:
        """
        Calculate validation score (0-100).

        Formula:
        - Each error: -10 points
        - Each warning: -2 points
        - Start at 100
        """
        score = 100
        score -= len(errors) * 10
        score -= len(warnings) * 2
        return max(0, score)
```

**Success Metrics:**
- Validation accuracy: 100% (no false negatives)
- Validation time: <1 second per agent output
- Coverage: 100% of regulatory requirements validated

---

### 2. Certification Framework

**Deliverables:**

| Component | Description | Phase |
|-----------|-------------|-------|
| **Certification Criteria** | Detailed criteria for each regulation | Phase 1 |
| **Certification Process** | Workflow for certifying agents | Phase 1 |
| **Certification Dashboard** | UI for tracking certification status | Phase 2 |
| **Auditor Interface** | UI for third-party auditors | Phase 3 |

**Technical Specifications:**

**Certification Criteria (CBAM Example):**
```yaml
certification:
  regulation: "CBAM Regulation 2023/956"
  version: "1.0.0"
  effective_date: "2025-12-30"

  criteria:
    - id: "CBAM-001"
      category: "Data Quality"
      requirement: "CN codes must be from official EU Combined Nomenclature"
      validation: "All CN codes validated against EU TARIC database"
      severity: "critical"

    - id: "CBAM-002"
      category: "Calculation Accuracy"
      requirement: "Embedded emissions calculated per CBAM Annex IV methodology"
      validation: "Calculation reviewed by certified climate scientist"
      severity: "critical"

    - id: "CBAM-003"
      category: "Emission Factors"
      requirement: "Emission factors from authoritative sources (IEA, IPCC)"
      validation: "All emission factors traced to authoritative database"
      severity: "critical"

    - id: "CBAM-004"
      category: "Audit Trail"
      requirement: "Complete provenance tracking (SHA-256 hashes)"
      validation: "All calculations have SHA-256 hash for audit trail"
      severity: "critical"

    - id: "CBAM-005"
      category: "JSON Format"
      requirement: "Output JSON matches EU CBAM Transitional Registry schema"
      validation: "JSON validated against official EU schema"
      severity: "critical"

    - id: "CBAM-006"
      category: "Documentation"
      requirement: "User documentation and methodology guide provided"
      validation: "Documentation reviewed and approved"
      severity: "high"

  pass_criteria:
    critical_failures: 0
    high_failures: 0
    medium_failures: "<3"
    low_failures: "<5"
```

**Certification Workflow:**
```
┌────────────────────────────────────────────────────────┐
│              Agent Certification Process                │
└────────────────────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │    Step 1: Automated Validation    │
        │  • Run golden test suite (1,000+)  │
        │  • Execute validation hooks        │
        │  • Security scan (Bandit)          │
        │  • Performance benchmark           │
        └────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │   Step 2: Manual Review (if needed)│
        │  • Climate scientist reviews edge  │
        │    cases and warnings              │
        │  • Policy analyst confirms regs    │
        │  • Quality score: pass/fail        │
        └────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │     Step 3: Certification Sign-Off │
        │  • Tech Lead approves              │
        │  • Certificate generated (PDF)     │
        │  • Agent marked "certified"        │
        │  • Published to registry           │
        └────────────────────────────────────┘
```

**Certification Certificate Template:**
```markdown
# GreenLang Agent Certification

**Agent ID:** GL-CBAM-APP
**Version:** 1.0.0
**Regulation:** CBAM Regulation 2023/956
**Certification Date:** 2025-12-03
**Certification ID:** CERT-CBAM-001

---

## Certification Summary

This agent has been certified by the GreenLang Climate Science & Policy Team as **compliant** with CBAM Regulation 2023/956 (Carbon Border Adjustment Mechanism).

**Certification Status:** PASSED

**Validation Results:**
- Golden Tests Passed: 1,000 / 1,000 (100%)
- Validation Hooks Passed: 25 / 25 (100%)
- Security Scan: No critical vulnerabilities
- Performance: <10 minutes for 10,000 shipments

---

## Criteria Met

| ID | Criterion | Status | Evidence |
|----|-----------|--------|----------|
| CBAM-001 | CN codes from EU TARIC | PASS | All 100+ CN codes validated |
| CBAM-002 | Annex IV methodology | PASS | Reviewed by climate scientist |
| CBAM-003 | Authoritative emission factors | PASS | IEA, IPCC sources verified |
| CBAM-004 | SHA-256 provenance | PASS | All calculations hashed |
| CBAM-005 | EU JSON schema | PASS | Schema validation passed |
| CBAM-006 | Documentation | PASS | User guide approved |

---

## Certification Team

**Climate Science Lead:** Dr. Jane Smith
**Policy Analyst:** John Doe
**Certification Date:** 2025-12-03

**Digital Signature:** [SHA-256 hash of certification]

---

## Validity

This certification is valid until:
- **Expiration Date:** 2026-12-03 (12 months)
- **Or until:** CBAM Regulation 2023/956 is amended

**Re-Certification Required:** Annually or upon regulation change
```

**Success Metrics:**
- Certification pass rate on first attempt: >90%
- Certification time: <2 days
- Audit readiness: 100% (all certified agents audit-ready)

---

### 3. Golden Test Suites

**Deliverables:**

| Component | Description | Phase |
|-----------|-------------|-------|
| **CBAM Golden Tests** | 1,000+ test cases for CBAM calculations | Phase 1 |
| **EUDR Golden Tests** | 500+ test cases for EUDR due diligence | Phase 2 |
| **CSRD Golden Tests** | 500+ test cases for CSRD disclosures | Phase 2 |
| **Test Data Generator** | Tool for generating synthetic test data | Phase 3 |

**Technical Specifications:**

**Golden Test Case Structure:**
```yaml
# File: tests/golden/cbam/test_steel_china_001.yaml

test_case:
  id: "cbam_steel_china_001"
  regulation: "CBAM Regulation 2023/956"
  category: "steel"
  subcategory: "hot_rolled_coil"

  metadata:
    created_by: "dr_jane_smith"
    created_date: "2025-11-15"
    reviewed_by: "john_doe"
    reviewed_date: "2025-11-20"
    confidence: "high"  # high, medium, low
    tags: ["steel", "china", "hot_rolled", "basic_calculation"]

  input:
    cn_code: "7208.10.00"  # Hot-rolled products of iron or steel
    product_name: "Hot-rolled steel coil"
    origin_country: "CN"  # China
    weight_kg: 10000
    production_route: "blast_furnace_basic_oxygen_furnace"

  expected_output:
    emission_factor_tco2_per_tonne: 2.1  # IEA 2024 database
    emission_factor_source: "IEA Steel Technology Roadmap 2024"
    emission_factor_reference_url: "https://iea.org/reports/steel-2024"

    embedded_emissions_tco2e: 21.0  # 10 tonnes × 2.1 tCO2/tonne

    calculation_steps:
      - step: "Convert weight to tonnes"
        formula: "10000 kg ÷ 1000 = 10 tonnes"
        result: 10

      - step: "Lookup emission factor"
        formula: "IEA database: China, BF-BOF route → 2.1 tCO2/tonne"
        result: 2.1

      - step: "Calculate embedded emissions"
        formula: "10 tonnes × 2.1 tCO2/tonne = 21.0 tCO2e"
        result: 21.0

    provenance:
      data_sources: ["IEA Steel Technology Roadmap 2024", "EU TARIC CN codes"]
      calculation_methodology: "CBAM Regulation 2023/956 Annex IV"
      validation_date: "2025-11-20"

  validation:
    tolerance: 0.01  # Allow 1% rounding error
    expected_range:
      min: 20.0  # -5% tolerance
      max: 22.0  # +5% tolerance

  edge_cases:
    - description: "Missing production route should use country average"
      input_override: {production_route: null}
      expected_emission_factor: 2.3  # China average (all routes)

    - description: "Unknown subcategory should use category average"
      input_override: {cn_code: "7208.99.00"}
      expected_emission_factor: 2.2  # Steel category average
```

**Test Suite Execution:**
```python
class GoldenTestRunner:
    """Execute golden test suites."""

    def run_suite(self, suite_name: str, agent: Agent) -> TestSuiteResult:
        """
        Run all tests in suite.

        Args:
            suite_name: Name of test suite (e.g., "cbam_steel")
            agent: Agent under test

        Returns:
            TestSuiteResult with pass/fail for each test
        """
        tests = self.load_tests(suite_name)
        results = []

        for test in tests:
            result = self.run_test(test, agent)
            results.append(result)

        return TestSuiteResult(
            suite_name=suite_name,
            total_tests=len(tests),
            passed=sum(1 for r in results if r.passed),
            failed=sum(1 for r in results if not r.passed),
            pass_rate=sum(1 for r in results if r.passed) / len(tests),
            results=results
        )

    def run_test(self, test: GoldenTest, agent: Agent) -> TestResult:
        """Run single golden test."""
        # Execute agent
        agent_output = agent.execute(test.input)

        # Compare with expected output
        is_match = self.compare_output(
            expected=test.expected_output,
            actual=agent_output,
            tolerance=test.validation.tolerance
        )

        return TestResult(
            test_id=test.id,
            passed=is_match,
            expected=test.expected_output,
            actual=agent_output,
            diff=self.generate_diff(test.expected_output, agent_output)
        )
```

**Success Metrics:**
- Golden test coverage: 1,000+ tests by Phase 1, 2,000+ by Phase 3
- Test pass rate: >95% for certified agents
- Test maintenance: <10% annual test updates (regulations stable)

---

### 4. Regulatory Intelligence (Monitoring & Updates)

**Deliverables:**

| Component | Description | Phase |
|-----------|-------------|-------|
| **Regulation Tracker** | Monitor EU Official Journal for updates | Phase 1 |
| **Impact Assessment** | Assess impact of regulation changes on agents | Phase 2 |
| **Update Notifications** | Alert teams when regulations change | Phase 2 |
| **Migration Guides** | Guides for updating agents to new regulations | Phase 3 |

**Technical Specifications:**

**Regulation Monitoring Sources:**
- EU Official Journal (EUR-Lex)
- European Commission climate policy updates
- CBAM Transitional Registry technical updates
- EUDR implementing acts
- CSRD delegated acts

**Monitoring Workflow:**
```
┌────────────────────────────────────────────────────────┐
│          Regulatory Change Management                  │
└────────────────────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │  Step 1: Monitor Sources (Weekly)  │
        │  • EUR-Lex RSS feeds               │
        │  • EC climate policy page          │
        │  • Industry newsletters            │
        └────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │  Step 2: Impact Assessment         │
        │  • Identify affected agents        │
        │  • Estimate update effort          │
        │  • Prioritize by deadline          │
        └────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │  Step 3: Update Plan               │
        │  • Create migration guide          │
        │  • Update validation hooks         │
        │  • Update golden tests             │
        │  • Notify affected customers       │
        └────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │  Step 4: Re-Certification          │
        │  • Re-certify affected agents      │
        │  • Issue updated certificates      │
        │  • Publish to registry             │
        └────────────────────────────────────┘
```

**Success Metrics:**
- Regulation update detection: <7 days from publication
- Impact assessment: <14 days
- Agent updates: <30 days from regulation effective date

---

## Deliverables by Phase

### Phase 1: Foundation (Weeks 1-16)

**Milestone:** CBAM validation framework operational

**Week 1-4: Validation Hooks (CBAM)**
- [ ] CBAM validation hook implementation
- [ ] Emission factor database (100+ factors from IEA, IPCC)
- [ ] CN code validator (EU TARIC database)
- [ ] Validation SDK core library

**Week 5-8: Golden Tests (CBAM)**
- [ ] 100 CBAM golden test cases (steel, cement, aluminum)
- [ ] Test runner framework
- [ ] Test data generator
- [ ] Test documentation

**Week 9-12: Certification Framework**
- [ ] CBAM certification criteria
- [ ] Certification workflow
- [ ] Certification certificate template
- [ ] Manual review process

**Week 13-16: Regulatory Intelligence**
- [ ] EUR-Lex monitoring setup
- [ ] Regulation tracker dashboard
- [ ] Update notification system
- [ ] Impact assessment framework

**Phase 1 Exit Criteria:**
- [ ] CBAM validation hooks operational
- [ ] 100+ CBAM golden tests
- [ ] Certification framework live
- [ ] GL-CBAM-APP certified
- [ ] Regulation monitoring active

---

### Phase 2: Production Scale (Weeks 17-28)

**Milestone:** Multi-regulation validation framework

**Week 17-20: EUDR Validation**
- [ ] EUDR validation hooks
- [ ] Geolocation validator
- [ ] Due diligence checker
- [ ] 200+ EUDR golden tests

**Week 21-24: CSRD Validation**
- [ ] CSRD validation hooks
- [ ] Double materiality validator
- [ ] ESG metrics checker
- [ ] 200+ CSRD golden tests

**Week 25-28: Test Suite Expansion**
- [ ] 1,000+ CBAM golden tests
- [ ] 500+ EUDR golden tests
- [ ] 500+ CSRD golden tests
- [ ] Automated test generation

**Phase 2 Exit Criteria:**
- [ ] EUDR and CSRD validation hooks operational
- [ ] 2,000+ total golden tests
- [ ] 10 agents certified
- [ ] Test pass rate: >95%

---

### Phase 3: Enterprise Ready (Weeks 29-40)

**Milestone:** Enterprise-grade certification and auditing

**Week 29-32: Auditor Interface**
- [ ] Third-party auditor UI
- [ ] Audit trail viewer
- [ ] Certificate verification tool
- [ ] Audit report generator

**Week 33-36: Advanced Validation**
- [ ] Machine learning for anomaly detection
- [ ] Automated edge case discovery
- [ ] Continuous monitoring of agent outputs
- [ ] Real-time validation alerts

**Week 37-40: Scale & Automation**
- [ ] Batch certification (100+ agents)
- [ ] Automated re-certification on updates
- [ ] Migration tooling for regulation changes
- [ ] Enterprise documentation

**Phase 3 Exit Criteria:**
- [ ] Auditor interface operational
- [ ] 100 agents certified
- [ ] Automated re-certification
- [ ] Enterprise audit compliance

---

## Success Metrics & KPIs

### North Star Metrics

| Metric | Phase 1 Target | Phase 2 Target | Phase 3 Target | Measurement |
|--------|---------------|---------------|---------------|-------------|
| **Regulatory Compliance** | 100% | 100% | 100% | % of certified agents compliant |
| **Certification Pass Rate** | >90% | >95% | >98% | % passing on first attempt |
| **Golden Test Coverage** | 100 tests | 2,000 tests | 5,000 tests | Number of test cases |
| **Validation Accuracy** | 100% | 100% | 100% | % correct validations (no false negatives) |
| **Audit Readiness** | 100% | 100% | 100% | % of agents audit-ready |

### Team Performance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Regulation Update Detection** | <7 days | Time from publication to detection |
| **Impact Assessment** | <14 days | Time to assess impact of regulation change |
| **Certification Time** | <2 days | Time from request to certification |
| **Test Maintenance** | <10%/year | % of tests requiring updates annually |

---

## Interfaces with Other Teams

### AI/Agent Team

**What Climate Science Provides:**
- Validation hooks for agent outputs
- Golden test cases
- Certification framework

**What Climate Science Receives:**
- Generated agent code for review
- Agent SDK integration points
- Feedback on validation performance

**Integration Points:**
- Agent SDK calls validation hooks
- AI/Agent integrates golden tests into CI/CD

**Meeting Cadence:**
- Weekly: Review generated agents
- Bi-Weekly: Validation framework sync

---

### ML Platform Team

**What Climate Science Provides:**
- Domain-specific test cases
- Validation rules
- Feedback on model accuracy

**What Climate Science Receives:**
- Evaluation framework
- Golden test infrastructure
- Model outputs for review

**Integration Points:**
- Climate Science contributes golden tests to ML Platform
- ML Platform runs validation hooks

**Meeting Cadence:**
- Weekly: Review new test cases
- Monthly: Model quality review

---

## Technical Stack

### Languages & Tools

- **Python 3.11+** (validation hooks)
- **YAML** (test case definitions)
- **Pydantic** (data validation)
- **Pytest** (test execution)

### Data Sources

- **IEA (International Energy Agency):** Emission factors
- **IPCC (Intergovernmental Panel on Climate Change):** GHG methodologies
- **EU TARIC:** CN codes database
- **EUR-Lex:** Regulatory texts

---

## Risks & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Regulation changes invalidate agents** | Medium | High | Continuous monitoring; modular validation hooks |
| **Emission factor database outdated** | Medium | High | Quarterly updates from IEA/IPCC |
| **Certification bottleneck** | Low | Medium | Automated validation; manual review only for edge cases |
| **Test suite gaps** | Medium | High | Continuous test expansion; human review |

---

**Document Control:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-03 | GL Product Manager | Initial Climate Science Team charter |

---

**Approvals:**

- Climate Science Tech Lead: ___________________
- Engineering Lead: ___________________
- Product Manager: ___________________
