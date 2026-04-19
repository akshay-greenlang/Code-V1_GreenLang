# Phase 2: Factory Core - Generator and Evaluation

**Version:** 1.0
**Date:** 2025-12-03
**Product Manager:** GL-ProductManager
**Phase Duration:** 12 weeks (Mar 1 - May 23, 2026)
**Status:** Planned

---

## Executive Summary

Phase 2 is the transformational phase that moves GreenLang from "write agents by hand" to "generate from spec." This phase delivers the Agent Generator (spec-to-code), the Evaluation Framework (quality assurance), and the Certification Pipeline (regulatory validation).

**Phase Goal:** Enable automated generation of production-quality agents from AgentSpec v2 with rigorous evaluation and certification.

**Key Outcome:** 10+ agents generated from spec, certified, and deployed to production with <2 hours per agent (vs. 2 weeks manual).

---

## Objectives

### Primary Objectives

1. **Finalize AgentSpec v2** - Extended schema with generation metadata
2. **Build Agent Generator** - Transforms AgentSpec into executable agent code
3. **Create Evaluation Framework** - Automated quality assessment
4. **Implement Certification Pipeline** - Domain validation and approval workflow
5. **Generate 10 Agents** - Prove factory capability at scale

### Non-Objectives (Out of Scope for Phase 2)

- Agent Registry and discovery (Phase 3)
- Runtime governance (Phase 3)
- Self-service UI (Phase 4)
- External partner access (Phase 4)

---

## Technical Scope

### Component 1: AgentSpec v2

**Description:** Extended AgentSpec with generation hints, prompt templates, and evaluation criteria.

**New Fields in v2:**

```yaml
# AgentSpec v2 Extensions (in addition to v1 fields)
generation:
  model_preferences:
    primary: claude-3-opus
    fallback: claude-3-sonnet
    temperature: 0.0
    max_tokens: 4096

  prompt_templates:
    - name: main_prompt
      template_file: prompts/main.jinja2
      variables:
        - name: input_schema
          source: inputs
        - name: output_schema
          source: outputs

    - name: tool_selection_prompt
      template_file: prompts/tool_selection.jinja2

  code_generation:
    language: python
    sdk_version: ">=1.0.0"
    style_guide: google
    docstring_format: google

evaluation:
  golden_tests:
    - name: basic_calculation
      input_file: tests/inputs/basic.json
      expected_output_file: tests/outputs/basic.json
      tolerance: 0.01

    - name: edge_case_missing_data
      input_file: tests/inputs/missing_data.json
      expected_behavior: fallback_to_default

  benchmarks:
    - name: accuracy
      metric: exact_match
      threshold: 0.95

    - name: latency
      metric: p99_ms
      threshold: 5000

    - name: cost
      metric: tokens_per_request
      threshold: 10000

  domain_validation:
    regulations: [CSRD, CBAM]
    expert_review_required: true
    golden_test_count: 50

certification:
  required_approvals:
    - role: climate_scientist
      count: 2
    - role: domain_expert
      count: 1

  compliance_checks:
    - csrd_esrs_alignment
    - cbam_schema_validation
    - data_provenance_verification

  deployment_gates:
    - test_coverage: 85
    - security_scan: pass
    - performance_benchmark: pass
```

**Deliverables:**
- AgentSpec v2 JSON Schema
- Migration guide from v1 to v2
- Prompt template library (10+ templates)
- Documentation with generation examples

**Owner:** AI/Agent Team
**Support:** ML Platform (evaluation fields), Climate Science (certification fields)

---

### Component 2: Agent Generator

**Description:** Core component that transforms AgentSpec v2 into executable Python agent code.

**Generator Architecture:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Agent Generator                              │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  AgentSpec   │───►│   Parser &   │───►│   Code Generator     │  │
│  │  v2 Input    │    │   Validator  │    │                      │  │
│  └──────────────┘    └──────────────┘    │  ┌────────────────┐  │  │
│                                          │  │ Class Template │  │  │
│                                          │  └────────────────┘  │  │
│                                          │  ┌────────────────┐  │  │
│                                          │  │ Prompt Builder │  │  │
│                                          │  └────────────────┘  │  │
│                                          │  ┌────────────────┐  │  │
│                                          │  │ Test Generator │  │  │
│                                          │  └────────────────┘  │  │
│                                          └──────────────────────┘  │
│                                                     │              │
│                                                     ▼              │
│                                          ┌──────────────────────┐  │
│                                          │   Output Bundle      │  │
│                                          │  ┌────────────────┐  │  │
│                                          │  │ agent.py       │  │  │
│                                          │  │ prompts/       │  │  │
│                                          │  │ tests/         │  │  │
│                                          │  │ config.yaml    │  │  │
│                                          │  └────────────────┘  │  │
│                                          └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

**Generator Interface:**

```python
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel

class GeneratorConfig(BaseModel):
    """Configuration for agent generation."""
    output_dir: Path
    sdk_version: str = "1.0.0"
    include_tests: bool = True
    include_docs: bool = True
    dry_run: bool = False

class GeneratorResult(BaseModel):
    """Result of agent generation."""
    success: bool
    agent_id: str
    output_path: Path
    files_generated: list[str]
    warnings: list[str]
    errors: list[str]
    generation_time_ms: int
    token_usage: Dict[str, int]

class AgentGenerator:
    """Generates executable agents from AgentSpec v2."""

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self._template_engine = TemplateEngine()
        self._code_generator = CodeGenerator()
        self._test_generator = TestGenerator()

    async def generate(
        self,
        spec: AgentSpec,
        context: Optional[Dict[str, Any]] = None
    ) -> GeneratorResult:
        """Generate agent from spec."""
        start_time = time.time()
        errors = []
        warnings = []
        files = []

        # 1. Validate spec
        validation_result = await self._validate_spec(spec)
        if not validation_result.passed:
            return GeneratorResult(
                success=False,
                errors=validation_result.errors,
                ...
            )

        # 2. Generate prompts from templates
        prompts = await self._generate_prompts(spec)
        files.extend(self._write_prompts(prompts))

        # 3. Generate agent class code
        agent_code = await self._generate_agent_code(spec)
        files.append(self._write_agent_code(agent_code))

        # 4. Generate tests from golden test specs
        if self.config.include_tests:
            tests = await self._generate_tests(spec)
            files.extend(self._write_tests(tests))

        # 5. Generate configuration
        config = self._generate_config(spec)
        files.append(self._write_config(config))

        # 6. Generate documentation
        if self.config.include_docs:
            docs = self._generate_docs(spec)
            files.append(self._write_docs(docs))

        return GeneratorResult(
            success=True,
            agent_id=spec.agent_id,
            output_path=self.config.output_dir / spec.agent_id,
            files_generated=files,
            warnings=warnings,
            errors=[],
            generation_time_ms=int((time.time() - start_time) * 1000),
            token_usage=self._get_token_usage()
        )

    async def _generate_agent_code(self, spec: AgentSpec) -> str:
        """Generate Python agent class from spec."""
        # Uses code generation templates, NOT LLM
        # Deterministic, reproducible output
        return self._code_generator.generate(
            template="agent_class",
            spec=spec,
            imports=self._resolve_imports(spec),
            tools=self._resolve_tools(spec),
            validators=self._resolve_validators(spec)
        )

    async def _generate_prompts(self, spec: AgentSpec) -> Dict[str, str]:
        """Generate prompt files from templates."""
        prompts = {}
        for template_ref in spec.generation.prompt_templates:
            template = self._template_engine.load(template_ref.template_file)
            variables = self._resolve_variables(template_ref.variables, spec)
            prompts[template_ref.name] = template.render(**variables)
        return prompts
```

**Zero-Hallucination Architecture:**

The Generator uses deterministic templates, NOT LLM generation for code:

1. **Code Generation:** Jinja2 templates with parameterization
2. **Prompt Generation:** Template interpolation with spec values
3. **Test Generation:** Template-based test scaffolding
4. **NO LLM in Loop:** LLM only used at runtime by the agent, not during generation

**Deliverables:**
- `AgentGenerator` class with full implementation
- Code generation templates (agent class, tools, validators)
- Prompt template library
- Test generation templates
- CLI tool: `greenlang-generate agent --spec spec.yaml --output ./agents/`
- Documentation and examples

**Owner:** AI/Agent Team
**Support:** ML Platform (evaluation integration), Platform (CLI)

---

### Component 3: Evaluation Framework

**Description:** Automated system for evaluating generated agents against quality benchmarks.

**Evaluation Pipeline:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Evaluation Framework                           │
│                                                                     │
│  ┌──────────────┐                                                   │
│  │ Generated    │                                                   │
│  │ Agent        │                                                   │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Evaluation Stages                           │ │
│  │                                                                │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │ │
│  │  │   Stage 1   │  │   Stage 2   │  │   Stage 3   │            │ │
│  │  │   Unit      │──│   Golden    │──│   Domain    │            │ │
│  │  │   Tests     │  │   Tests     │  │   Validation│            │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘            │ │
│  │                                                                │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │ │
│  │  │   Stage 4   │  │   Stage 5   │  │   Stage 6   │            │ │
│  │  │   Perf      │──│   Security  │──│   Human     │            │ │
│  │  │   Benchmark │  │   Scan      │  │   Review    │            │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘            │ │
│  │                                                                │ │
│  └────────────────────────────────────────────────────────────────┘ │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐                                                   │
│  │ Evaluation   │ ──► Pass/Fail + Score + Report                    │
│  │ Report       │                                                   │
│  └──────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

**Evaluation Stages:**

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pydantic import BaseModel
from enum import Enum

class EvaluationStage(str, Enum):
    UNIT_TESTS = "unit_tests"
    GOLDEN_TESTS = "golden_tests"
    DOMAIN_VALIDATION = "domain_validation"
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    SECURITY_SCAN = "security_scan"
    HUMAN_REVIEW = "human_review"

class StageResult(BaseModel):
    """Result of a single evaluation stage."""
    stage: EvaluationStage
    passed: bool
    score: float  # 0.0 to 1.0
    details: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    duration_ms: int

class EvaluationReport(BaseModel):
    """Complete evaluation report for an agent."""
    agent_id: str
    version: str
    timestamp: str
    overall_passed: bool
    overall_score: float
    stages: List[StageResult]
    certification_eligible: bool
    blockers: List[str]
    recommendations: List[str]

class BaseEvaluator(ABC):
    """Abstract base class for evaluation stages."""

    stage: EvaluationStage

    @abstractmethod
    async def evaluate(
        self,
        agent: BaseAgent,
        spec: AgentSpec,
        context: Dict[str, Any]
    ) -> StageResult:
        pass

class GoldenTestEvaluator(BaseEvaluator):
    """Runs golden tests against agent."""

    stage = EvaluationStage.GOLDEN_TESTS

    async def evaluate(
        self,
        agent: BaseAgent,
        spec: AgentSpec,
        context: Dict[str, Any]
    ) -> StageResult:
        passed_tests = 0
        failed_tests = []

        for test in spec.evaluation.golden_tests:
            input_data = self._load_test_input(test.input_file)
            expected = self._load_expected_output(test.expected_output_file)

            result = await agent.execute(input_data, context)

            if self._compare_outputs(result.outputs, expected, test.tolerance):
                passed_tests += 1
            else:
                failed_tests.append(test.name)

        total_tests = len(spec.evaluation.golden_tests)
        score = passed_tests / total_tests if total_tests > 0 else 0

        return StageResult(
            stage=self.stage,
            passed=score >= 0.95,  # 95% pass threshold
            score=score,
            details={
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests
            },
            errors=[f"Failed: {t}" for t in failed_tests],
            warnings=[],
            duration_ms=...
        )

class DomainValidationEvaluator(BaseEvaluator):
    """Validates domain-specific compliance."""

    stage = EvaluationStage.DOMAIN_VALIDATION

    async def evaluate(
        self,
        agent: BaseAgent,
        spec: AgentSpec,
        context: Dict[str, Any]
    ) -> StageResult:
        # Run domain validators from Climate Science
        validation_results = []
        for regulation in spec.evaluation.domain_validation.regulations:
            validator = self._get_domain_validator(regulation)
            result = await validator.validate(agent, spec)
            validation_results.append(result)

        all_passed = all(r.passed for r in validation_results)

        return StageResult(
            stage=self.stage,
            passed=all_passed,
            score=sum(r.score for r in validation_results) / len(validation_results),
            details={"regulation_results": validation_results},
            errors=[r.error for r in validation_results if not r.passed],
            warnings=[],
            duration_ms=...
        )

class EvaluationPipeline:
    """Orchestrates full evaluation pipeline."""

    def __init__(self):
        self._stages: List[BaseEvaluator] = [
            UnitTestEvaluator(),
            GoldenTestEvaluator(),
            DomainValidationEvaluator(),
            PerformanceBenchmarkEvaluator(),
            SecurityScanEvaluator()
        ]

    async def evaluate(
        self,
        agent: BaseAgent,
        spec: AgentSpec
    ) -> EvaluationReport:
        stage_results = []

        for evaluator in self._stages:
            result = await evaluator.evaluate(agent, spec, {})
            stage_results.append(result)

            # Stop on critical failure
            if not result.passed and evaluator.stage in [
                EvaluationStage.UNIT_TESTS,
                EvaluationStage.SECURITY_SCAN
            ]:
                break

        overall_passed = all(r.passed for r in stage_results)
        overall_score = sum(r.score for r in stage_results) / len(stage_results)

        return EvaluationReport(
            agent_id=spec.agent_id,
            version=spec.version,
            timestamp=datetime.utcnow().isoformat(),
            overall_passed=overall_passed,
            overall_score=overall_score,
            stages=stage_results,
            certification_eligible=overall_passed and overall_score >= 0.9,
            blockers=self._identify_blockers(stage_results),
            recommendations=self._generate_recommendations(stage_results)
        )
```

**Deliverables:**
- `EvaluationPipeline` class with all stages
- Golden test runner
- Performance benchmarking tools
- Security scanner integration (Snyk/Bandit)
- Evaluation report generator (HTML, JSON, PDF)
- CLI tool: `greenlang-evaluate agent --path ./agents/gl-cbam-v1/`

**Owner:** ML Platform Team
**Support:** Climate Science (domain validators), DevOps (security scan)

---

### Component 4: Certification Pipeline

**Description:** Workflow for certifying agents as production-ready after evaluation.

**Certification Workflow:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Certification Pipeline                          │
│                                                                     │
│  ┌──────────────┐                                                   │
│  │ Evaluation   │                                                   │
│  │ Report       │                                                   │
│  │ (Passed)     │                                                   │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                 Certification Steps                            │ │
│  │                                                                │ │
│  │  ┌─────────────────┐                                           │ │
│  │  │ 1. Auto-Checks  │ ──► Regulatory schema validation          │ │
│  │  │                 │ ──► Data provenance verification          │ │
│  │  │                 │ ──► Calculation audit                     │ │
│  │  └────────┬────────┘                                           │ │
│  │           │                                                    │ │
│  │           ▼                                                    │ │
│  │  ┌─────────────────┐                                           │ │
│  │  │ 2. Expert       │ ──► Climate Scientist review (2 required) │ │
│  │  │    Review       │ ──► Domain Expert review (1 required)     │ │
│  │  │                 │ ──► Regulatory Advisor review (optional)  │ │
│  │  └────────┬────────┘                                           │ │
│  │           │                                                    │ │
│  │           ▼                                                    │ │
│  │  ┌─────────────────┐                                           │ │
│  │  │ 3. Approval     │ ──► Collect approvals                     │ │
│  │  │    Workflow     │ ──► Record in audit log                   │ │
│  │  │                 │ ──► Generate certificate                  │ │
│  │  └────────┬────────┘                                           │ │
│  │           │                                                    │ │
│  └───────────┼────────────────────────────────────────────────────┘ │
│              │                                                      │
│              ▼                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Certified Agent                           │   │
│  │  ┌────────────────────────────────────────────────────────┐  │   │
│  │  │ Certificate ID: CERT-GL-CBAM-V1-20260315              │  │   │
│  │  │ Agent: gl-cbam-calculator-v1                          │  │   │
│  │  │ Version: 1.0.0                                        │  │   │
│  │  │ Certified By: [Climate Scientist 1, 2, Domain Expert] │  │   │
│  │  │ Valid Until: 2027-03-15                               │  │   │
│  │  │ Regulations: CBAM                                     │  │   │
│  │  └────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

**Certificate Data Model:**

```python
class CertificationStatus(str, Enum):
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    REVOKED = "revoked"

class ReviewerApproval(BaseModel):
    reviewer_id: str
    reviewer_name: str
    role: str  # climate_scientist, domain_expert, regulatory_advisor
    approved: bool
    comments: Optional[str]
    timestamp: datetime

class Certificate(BaseModel):
    certificate_id: str
    agent_id: str
    agent_version: str
    status: CertificationStatus
    evaluation_report_id: str
    regulatory_scope: List[str]
    approvals: List[ReviewerApproval]
    issued_at: Optional[datetime]
    expires_at: Optional[datetime]
    issued_by: Optional[str]
    revocation_reason: Optional[str]

class CertificationPipeline:
    """Manages agent certification workflow."""

    async def submit_for_certification(
        self,
        agent_id: str,
        version: str,
        evaluation_report: EvaluationReport
    ) -> Certificate:
        """Submit evaluated agent for certification."""
        if not evaluation_report.certification_eligible:
            raise CertificationError("Agent not eligible for certification")

        certificate = Certificate(
            certificate_id=self._generate_certificate_id(agent_id, version),
            agent_id=agent_id,
            agent_version=version,
            status=CertificationStatus.PENDING,
            evaluation_report_id=evaluation_report.id,
            regulatory_scope=self._extract_regulatory_scope(evaluation_report),
            approvals=[]
        )

        await self._run_auto_checks(certificate)
        await self._notify_reviewers(certificate)

        return certificate

    async def record_approval(
        self,
        certificate_id: str,
        approval: ReviewerApproval
    ) -> Certificate:
        """Record reviewer approval or rejection."""
        certificate = await self._get_certificate(certificate_id)
        certificate.approvals.append(approval)

        if self._has_required_approvals(certificate):
            certificate.status = CertificationStatus.APPROVED
            certificate.issued_at = datetime.utcnow()
            certificate.expires_at = certificate.issued_at + timedelta(days=365)
            await self._issue_certificate(certificate)
        elif any(not a.approved for a in certificate.approvals):
            certificate.status = CertificationStatus.REJECTED

        return certificate
```

**Deliverables:**
- `CertificationPipeline` class
- Certificate data model and storage
- Reviewer notification system
- Approval workflow UI (or API)
- Certificate generation (PDF, JSON)
- Audit log for all certification actions

**Owner:** Climate Science Team
**Support:** DevOps (workflow automation), Platform (storage)

---

### Component 5: Golden Test Suite

**Description:** Comprehensive test suite created by Climate Science to validate agent correctness.

**Golden Test Categories:**

| Category | Count | Purpose |
|----------|-------|---------|
| Basic Calculations | 30 | Core arithmetic and formula verification |
| Edge Cases | 25 | Missing data, invalid inputs, boundary conditions |
| Regulatory Scenarios | 30 | CSRD, CBAM, EUDR compliance |
| Multi-Step Workflows | 15 | Complex agent flows |
| **Total** | **100** | Minimum for certification |

**Golden Test Format:**

```yaml
# Golden Test Definition
test_id: gt-cbam-001
name: Basic Steel Import Calculation
description: |
  Verify correct emissions calculation for steel imports
  from China with known emission factors.

agent_id: gl-cbam-calculator-v1
category: basic_calculations

input:
  imports:
    - product_category: steel
      cn_code: "72061000"
      origin_country: CN
      quantity_kg: 10000
  reporting_period:
    start_date: "2025-01-01"
    end_date: "2025-03-31"

expected_output:
  embedded_emissions:
    total_tco2e: 19.5  # 10 tonnes * 1.95 tCO2e/tonne (China steel)
    by_product:
      steel: 19.5
    by_country:
      CN: 19.5

tolerance: 0.01  # 1% tolerance for floating point

validation:
  - type: arithmetic
    check: total_equals_sum_of_parts
  - type: provenance
    check: emission_factor_source_is_iea

metadata:
  created_by: climate_science_team
  created_at: "2025-12-03"
  regulation: CBAM
  reference: EU CBAM Regulation 2023/956
```

**Deliverables:**
- 100+ golden tests for flagship agents
- Golden test schema and validator
- Test data generator for edge cases
- Golden test documentation

**Owner:** Climate Science Team
**Support:** AI/Agent (integration), ML Platform (runner)

---

## Deliverables by Team

### AI/Agent Team (Primary Owner)

| Deliverable | Effort | Due | Status |
|-------------|--------|-----|--------|
| AgentSpec v2 schema | 2 weeks | Week 14 | Pending |
| AgentSpec v2 migration guide | 1 week | Week 15 | Pending |
| Agent Generator core | 4 weeks | Week 18 | Pending |
| Prompt template library | 2 weeks | Week 16 | Pending |
| Code generation templates | 2 weeks | Week 18 | Pending |
| Generate 10 agents | 4 weeks | Week 24 | Pending |
| Generator CLI tool | 1 week | Week 20 | Pending |
| Generator documentation | 2 weeks | Week 24 | Pending |

### ML Platform Team (Primary Owner - Evaluation)

| Deliverable | Effort | Due | Status |
|-------------|--------|-----|--------|
| Evaluation pipeline core | 3 weeks | Week 18 | Pending |
| Golden test evaluator | 2 weeks | Week 16 | Pending |
| Performance benchmark evaluator | 2 weeks | Week 18 | Pending |
| Security scan integration | 1 week | Week 20 | Pending |
| Evaluation report generator | 2 weeks | Week 22 | Pending |
| Evaluation CLI tool | 1 week | Week 22 | Pending |

### Climate Science Team (Primary Owner - Certification)

| Deliverable | Effort | Due | Status |
|-------------|--------|-----|--------|
| Golden test suite (100+ tests) | 4 weeks | Week 20 | Pending |
| Domain validators (CSRD, CBAM) | 3 weeks | Week 18 | Pending |
| Certification workflow design | 1 week | Week 14 | Pending |
| Certification pipeline | 2 weeks | Week 18 | Pending |
| Expert review process | 2 weeks | Week 20 | Pending |
| Certify 10 agents | 4 weeks | Week 24 | Pending |

### Platform Team (Support)

| Deliverable | Effort | Due | Status |
|-------------|--------|-----|--------|
| Certificate storage | 1 week | Week 16 | Pending |
| Approval workflow API | 2 weeks | Week 18 | Pending |
| Audit log infrastructure | 1 week | Week 18 | Pending |

### DevOps Team (Support)

| Deliverable | Effort | Due | Status |
|-------------|--------|-----|--------|
| CI/CD for generator | 2 weeks | Week 16 | Pending |
| Evaluation infrastructure | 2 weeks | Week 18 | Pending |
| Security scanning pipeline | 1 week | Week 18 | Pending |

---

## Timeline

### Sprint Breakdown (2-Week Sprints)

**Sprint 7 (Weeks 13-14): Foundation**
- AgentSpec v2 schema design and implementation
- Evaluation pipeline architecture design
- Certification workflow design
- Golden test framework setup

**Sprint 8 (Weeks 15-16): Core Components**
- Agent Generator core implementation
- Golden test evaluator implementation
- Prompt template library
- Certificate storage

**Sprint 9 (Weeks 17-18): Integration**
- Agent Generator completion
- Evaluation pipeline completion
- Certification pipeline implementation
- Domain validators (CSRD, CBAM)

**Sprint 10 (Weeks 19-20): First Agents**
- Generate first 5 agents
- Golden test suite completion
- Evaluation and certification for first agents
- CLI tools

**Sprint 11 (Weeks 21-22): Scale**
- Generate agents 6-8
- Evaluation report generator
- Expert review process
- Documentation

**Sprint 12 (Weeks 23-24): Polish**
- Generate agents 9-10
- Certify all 10 agents
- Final testing and bug fixes
- Phase 2 exit review preparation

---

## Success Criteria

### Must-Have (Phase Cannot Exit Without)

| Criteria | Target | Measurement |
|----------|--------|-------------|
| AgentSpec v2 complete | 100% | Schema passes validation |
| Generator produces valid code | 90%+ | % of specs that generate executable agents |
| Evaluation pipeline operational | 100% | All stages running |
| Certification pipeline operational | 100% | Workflow complete |
| Agents generated | 10+ | From AgentSpec v2 |
| Agents certified | 10+ | Passing all certification |
| Golden tests | 100+ | Climate Science approved |
| Certification pass rate | 90%+ | First-attempt pass rate |
| Generation time | <2 hours | Per agent, end-to-end |

### Should-Have

| Criteria | Target | Measurement |
|----------|--------|-------------|
| CLI tools | 2 | Generator + Evaluation |
| Documentation complete | 100% | User guides, API docs |
| Evaluation report formats | 3 | HTML, JSON, PDF |
| Domain validators | 2+ | CSRD, CBAM |

### Could-Have

| Criteria | Target | Measurement |
|----------|--------|-------------|
| EUDR domain validator | 1 | Passing tests |
| Batch generation | 1 | Generate multiple agents |
| Certification dashboard | 1 | View status |

---

## Exit Criteria to Phase 3

**All Must Pass to Proceed to Phase 3:**

1. **Generator Production-Ready**
   - [ ] 90%+ of specs generate valid, executable code
   - [ ] Generation time <2 hours per agent
   - [ ] Zero-hallucination verified (deterministic output)
   - [ ] Documentation complete

2. **Evaluation Framework Operational**
   - [ ] All 6 evaluation stages implemented
   - [ ] 95%+ accuracy on golden tests
   - [ ] Security scanning integrated
   - [ ] Reports generated automatically

3. **Certification Pipeline Active**
   - [ ] Workflow supports 2 climate scientist + 1 domain expert approval
   - [ ] Certificates issued with expiration
   - [ ] Audit log capturing all actions
   - [ ] 10 agents certified

4. **Golden Test Suite Complete**
   - [ ] 100+ golden tests created
   - [ ] All regulatory scenarios covered (CSRD, CBAM)
   - [ ] Edge cases documented
   - [ ] Climate Science sign-off

5. **Scale Validated**
   - [ ] 10+ agents generated and certified
   - [ ] Parallel generation tested
   - [ ] No performance regressions

**Phase 3 Kickoff Blocked If:**
- Generator success rate <80%
- Certification pass rate <80%
- Critical bugs in evaluation or certification
- Fewer than 8 agents certified

---

## Risks and Mitigations

### Phase 2 Specific Risks

| Risk | Likelihood | Impact | Owner | Mitigation |
|------|------------|--------|-------|------------|
| Generator quality issues | High | High | AI/Agent | Extensive templates; human review |
| Golden test creation delays | Medium | High | Climate Science | Start in Sprint 7; prioritize |
| Certification bottleneck | Medium | Medium | Climate Science | Train additional reviewers |
| Evaluation too slow | Low | Medium | ML Platform | Parallel execution; caching |
| AgentSpec v2 scope creep | Medium | Medium | AI/Agent | Timebox to 2 weeks |

### Mitigation Actions

1. **Generator Quality**
   - Template-based (not LLM-generated) code
   - Extensive unit tests for templates
   - Human review for first 5 agents

2. **Certification Scaling**
   - Train 4 climate scientists as reviewers
   - Define clear review criteria
   - SLA: 48 hours for review

3. **Performance Optimization**
   - Parallel evaluation stages
   - Cache evaluation results
   - Incremental evaluation for re-submissions

---

## Resource Allocation

### Team Allocation by Week

| Team | W13-14 | W15-16 | W17-18 | W19-20 | W21-22 | W23-24 | Total |
|------|--------|--------|--------|--------|--------|--------|-------|
| AI/Agent | 5 | 5 | 5 | 5 | 5 | 5 | 60 FTE-weeks |
| ML Platform | 4 | 4 | 4 | 3 | 3 | 2 | 40 FTE-weeks |
| Climate Science | 2 | 3 | 4 | 4 | 4 | 3 | 40 FTE-weeks |
| Platform | 1 | 2 | 2 | 1 | 1 | 1 | 16 FTE-weeks |
| DevOps | 2 | 2 | 2 | 1 | 1 | 1 | 18 FTE-weeks |
| **Total** | 14 | 16 | 17 | 14 | 14 | 12 | **174 FTE-weeks** |

---

## Appendices

### Appendix A: Generator Output Example

**Input (AgentSpec v2):**
```yaml
agent_id: gl-cbam-calculator-v2
name: CBAM Embedded Emissions Calculator
type: single-turn
# ... (full spec)
```

**Output (Generated Files):**
```
gl-cbam-calculator-v2/
├── agent.py           # Generated agent class
├── prompts/
│   └── main.jinja2    # Generated from template
├── tests/
│   ├── test_agent.py  # Unit tests
│   └── test_golden.py # Golden test integration
├── config.yaml        # Agent configuration
└── README.md          # Generated documentation
```

### Appendix B: Evaluation Report Example

```json
{
  "agent_id": "gl-cbam-calculator-v2",
  "version": "1.0.0",
  "timestamp": "2026-03-15T10:30:00Z",
  "overall_passed": true,
  "overall_score": 0.94,
  "stages": [
    {
      "stage": "unit_tests",
      "passed": true,
      "score": 0.92,
      "details": {"total": 50, "passed": 46, "failed": 4}
    },
    {
      "stage": "golden_tests",
      "passed": true,
      "score": 0.96,
      "details": {"total": 100, "passed": 96, "failed": 4}
    },
    {
      "stage": "domain_validation",
      "passed": true,
      "score": 1.0,
      "details": {"regulations": ["CBAM"], "all_passed": true}
    },
    {
      "stage": "performance_benchmark",
      "passed": true,
      "score": 0.88,
      "details": {"p99_ms": 2500, "threshold": 5000}
    },
    {
      "stage": "security_scan",
      "passed": true,
      "score": 1.0,
      "details": {"vulnerabilities": 0}
    }
  ],
  "certification_eligible": true,
  "blockers": [],
  "recommendations": [
    "Consider adding tests for edge case: missing CN code"
  ]
}
```

---

**Document Control:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-03 | GL-ProductManager | Initial Phase 2 plan |

---

**Approvals:**

- Product Manager: ___________________
- AI/Agent Lead: ___________________
- ML Platform Lead: ___________________
- Climate Science Lead: ___________________
- Engineering Lead: ___________________
