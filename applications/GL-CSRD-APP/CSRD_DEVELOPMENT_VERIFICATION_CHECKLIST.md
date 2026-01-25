# 🎯 CSRD APP DEVELOPMENT VERIFICATION CHECKLIST

**Using GL-CBAM-APP as the Benchmark for Excellence**

**Document Version:** 1.0
**Created:** 2025-10-18
**Purpose:** Ensure GL-CSRD-APP is built with the same quality, architecture, and rigor as GL-CBAM-APP
**Target Audience:** Development Team, QA, Technical Leads

---

## 📊 EXECUTIVE SUMMARY FOR AI EXPERT

### What Made CBAM Successful (Our Benchmark)

The CBAM Importer Copilot succeeded because of **7 critical architectural decisions**:

1. **Zero-Hallucination Architecture** - No LLM in calculation path
2. **3-Agent Pipeline Pattern** - Clear separation: Intake → Calculate → Report
3. **Tool-First Design** - Database lookups + Python arithmetic only
4. **Complete Provenance** - Every number traceable to source
5. **50+ Validation Rules** - Comprehensive compliance checking
6. **Multi-Format I/O** - CSV/JSON/Excel flexibility
7. **Performance Benchmarks** - <10 min for 10K records (20× faster than manual)

### CSRD vs CBAM: Critical Similarities

| Aspect | CBAM | CSRD | Similarity Level |
|--------|------|------|-----------------|
| **Regulatory Driver** | EU CBAM 2023/956 | EU CSRD 2023/2464 | ⭐⭐⭐⭐⭐ |
| **Zero-Hallucination Need** | 100% (border compliance) | 100% (audit-grade reporting) | ⭐⭐⭐⭐⭐ |
| **Multi-Agent Architecture** | 3 agents | 6 agents (2× scale) | ⭐⭐⭐⭐⭐ |
| **Provenance Required** | SHA256 audit trail | Third-party assurance trail | ⭐⭐⭐⭐⭐ |
| **Validation Rules** | 50+ rules | 200+ rules (4× scale) | ⭐⭐⭐⭐⭐ |
| **Data Complexity** | 30 CN codes, 14 factors | 1,082 ESRS data points, 500+ formulas | ⭐⭐⭐⭐ (CSRD 70× more complex) |
| **Output Format** | JSON + Markdown | XBRL + PDF + JSON | ⭐⭐⭐⭐ (CSRD adds XBRL complexity) |
| **Pipeline Pattern** | Intake → Calculate → Report | Intake → Materiality → Calculate → Aggregate → Report → Audit | ⭐⭐⭐⭐⭐ |

**Key Insight:** CSRD is **CBAM at 2-3× scale**, with the same core DNA but more complexity. The architecture MUST be identical; only the data volume and agent count differ.

---

## 🏗️ ARCHITECTURE VERIFICATION CHECKLIST

### ✅ CATEGORY 1: PROJECT STRUCTURE & FILES

**Principle:** Mirror CBAM's directory structure exactly, scaled up appropriately.

| # | Item | CBAM Reference | CSRD Status | Verification |
|---|------|---------------|-------------|--------------|
| 1.1 | **Root directory matches pattern** | `GL-CBAM-APP/CBAM-Importer-Copilot/` | `GL-CSRD-APP/CSRD-Reporting-Platform/` | ☐ Same naming convention (Product-Descriptor/) |
| 1.2 | **Agents directory** | `agents/` with 3 agents | `agents/` with 6 agents | ☐ Same structure, 2× agent count |
| 1.3 | **Each agent in separate file** | `shipment_intake_agent.py` (standalone) | `intake_agent.py` (standalone) | ☐ One file per agent, not bundled |
| 1.4 | **Data directory** | `data/cn_codes.json`, `emission_factors.py` | `data/esrs_data_points.json`, `emission_factors.py`, etc. | ☐ Same pattern, ESRS data instead of CN codes |
| 1.5 | **Schemas directory** | `schemas/*.schema.json` (3 files) | `schemas/*.schema.json` (6+ files) | ☐ JSON Schema validation for all data contracts |
| 1.6 | **Rules directory** | `rules/cbam_rules.yaml` (50+ rules) | `rules/esrs_compliance_rules.yaml` (200+ rules) | ☐ YAML format, same structure, 4× rules |
| 1.7 | **Specs directory** | `specs/*_spec.yaml` (3 agent specs) | `specs/*_spec.yaml` (6 agent specs) | ☐ One spec per agent, same YAML format |
| 1.8 | **Examples directory** | `examples/demo_*.csv` | `examples/demo_*.csv` | ☐ Demo data for testing |
| 1.9 | **Docs directory** | `docs/BUILD_JOURNEY.md`, `API_REFERENCE.md` | `docs/BUILD_JOURNEY.md`, `API_REFERENCE.md` | ☐ Same documentation structure |
| 1.10 | **CLI directory** | `cli/cbam_commands.py` | `cli/csrd_commands.py` | ☐ Same CLI pattern |
| 1.11 | **SDK directory** | `sdk/cbam_sdk.py` | `sdk/csrd_sdk.py` | ☐ Same SDK pattern |
| 1.12 | **Provenance directory** | `provenance/provenance_utils.py` | `provenance/provenance_utils.py` | ☐ EXACT same provenance system |
| 1.13 | **Tests directory** | `tests/test_*.py` (65+ tests) | `tests/test_*.py` (150+ tests) | ☐ 2-3× test coverage |
| 1.14 | **Main pipeline file** | `cbam_pipeline.py` | `csrd_pipeline.py` | ☐ Same orchestration pattern |
| 1.15 | **pack.yaml** | Comprehensive pack definition | Same structure | ☐ Mirror CBAM pack.yaml structure |
| 1.16 | **gl.yaml** | GreenLang metadata | Same structure | ☐ Mirror CBAM gl.yaml structure |
| 1.17 | **requirements.txt** | Python dependencies | Extended dependencies | ☐ Include all CBAM dependencies + XBRL/AI libs |
| 1.18 | **README.md** | Comprehensive user guide | Same quality | ☐ Mirror CBAM README structure and depth |
| 1.19 | **PROJECT_CHARTER.md** | Project definition | Exists | ☐ Same structure as CBAM charter |
| 1.20 | **setup.py** | Package setup | Exists | ☐ Same packaging approach |

**Status Indicator:** If <20/20 items ✅, STOP and fix structure first.

---

### ✅ CATEGORY 2: ZERO-HALLUCINATION ARCHITECTURE

**Principle:** CRITICAL - This is the #1 differentiator. NO LLM in calculation path.

| # | Item | CBAM Reference | CSRD Requirement | Verification |
|---|------|---------------|-----------------|--------------|
| 2.1 | **CalculatorAgent is deterministic** | EmissionsCalculatorAgent: database + arithmetic only | Same for ALL 500+ ESRS formulas | ☐ NO LLM in CalculatorAgent |
| 2.2 | **Emission factors from database** | `emission_factors.py` (14 factors, hardcoded) | `emission_factors.py` + additional factor databases | ☐ ALL factors from files, NEVER LLM |
| 2.3 | **Formulas are Python code** | Direct multiplication: `mass * factor` | YAML formulas → Python code | ☐ Formulas from YAML, executed as code |
| 2.4 | **NO estimation in calculations** | If data missing → ERROR (not estimate) | Same: missing data = ERROR | ☐ NO "fallback to estimate" logic |
| 2.5 | **Reproducibility guaranteed** | Same input → same output | Same guarantee | ☐ Unit tests prove reproducibility |
| 2.6 | **Audit trail for every calculation** | Provenance chain with SHA256 | Same provenance system | ☐ EVERY metric has provenance record |
| 2.7 | **Tool-First annotation** | `@deterministic` decorator (conceptual) | Same pattern | ☐ Clear markers for deterministic vs AI |
| 2.8 | **LLM usage ONLY for AI agents** | NO LLM in Intake, Calculator, Packager | LLM ONLY in MaterialityAgent | ☐ Clear separation: AI vs deterministic |
| 2.9 | **Confidence scores for AI outputs** | N/A (no AI in CBAM) | MaterialityAgent outputs confidence | ☐ All AI outputs have confidence scores |
| 2.10 | **Human review flags for AI** | N/A | "HUMAN_REVIEW_REQUIRED" flags | ☐ Clear flags for AI-generated content |

**Critical Failure:** If ANY calculation uses LLM, project FAILS regulatory requirements.

---

### ✅ CATEGORY 3: AGENT ARCHITECTURE

**Principle:** Each agent follows CBAM's pattern: clear input/output, tool-based, fully tested.

| # | Item | CBAM Pattern | CSRD Implementation | Verification |
|---|------|--------------|-------------------|--------------|
| 3.1 | **Agent count** | 3 agents (Intake, Calculator, Packager) | 6 agents (Intake, Materiality, Calculator, Aggregator, Reporting, Audit) | ☐ 6 distinct agents, not monolith |
| 3.2 | **Each agent = single file** | `shipment_intake_agent.py` (1,200 lines) | Similar size per agent | ☐ No >2,000 line files |
| 3.3 | **Agent class pattern** | `class ShipmentIntakeAgent` with `process()` method | Same pattern for all 6 | ☐ `class XAgent` with `process()` |
| 3.4 | **Clear inputs** | `process(shipments, cn_codes, suppliers)` | Documented inputs for each agent | ☐ Type hints for all inputs |
| 3.5 | **Clear outputs** | Returns `dict` with `metadata` + `results` | Same pattern | ☐ Consistent output structure |
| 3.6 | **Initialization with config** | `__init__(cn_codes_path, rules_path)` | Same pattern | ☐ All config via __init__, not globals |
| 3.7 | **Logging in every agent** | `logger.info()` throughout | Same | ☐ Structured logging (not print()) |
| 3.8 | **Error handling** | Try/except with custom errors | Same | ☐ Custom exception classes |
| 3.9 | **Performance metrics** | `processing_time_seconds`, `records_per_second` | Same | ☐ All agents emit performance metrics |
| 3.10 | **Tool calls documented** | Comments: "TOOL: database lookup" | Same | ☐ Clear tool vs LLM markers |
| 3.11 | **Pydantic models** | `ShipmentRecord`, `ValidationIssue` | Similar models for ESRS | ☐ Type-safe data models |
| 3.12 | **Agent specs match code** | `specs/shipment_intake_agent_spec.yaml` matches implementation | Same for all 6 agents | ☐ Specs are up-to-date |

**Quality Gate:** Each agent MUST pass unit tests >90% coverage before pipeline integration.

---

### ✅ CATEGORY 4: DATA & VALIDATION

**Principle:** CBAM uses reference data (CN codes, factors) + validation rules (YAML). CSRD scales this up.

| # | Item | CBAM Approach | CSRD Scaling | Verification |
|---|------|--------------|--------------|--------------|
| 4.1 | **Reference data files** | `cn_codes.json` (30 codes, 240 lines) | `esrs_data_points.json` (1,082 points) | ☐ JSON format, complete coverage |
| 4.2 | **Emission factors** | `emission_factors.py` (14 factors, 1,240 lines) | Multiple factor files (500+) | ☐ Hardcoded in Python, sourced |
| 4.3 | **Validation rules** | `cbam_rules.yaml` (50+ rules, 400 lines) | `esrs_compliance_rules.yaml` (200+ rules) | ☐ YAML format, executable rules |
| 4.4 | **JSON Schemas** | 3 schemas (shipment, supplier, output) | 6+ schemas (intake, materiality, metrics, etc.) | ☐ All data contracts in JSON Schema |
| 4.5 | **Rule engine** | Custom validator reads YAML rules | Same pattern | ☐ Rules are data, not hardcoded |
| 4.6 | **Error codes** | `E001-E010`, `W001-W005` in code | Extended error code system | ☐ Consistent error code pattern |
| 4.7 | **Data source citations** | `EMISSION_FACTORS_SOURCES.md` (450 lines) | Similar for ESRS factors | ☐ ALL factors have sources |
| 4.8 | **Demo data** | `demo_shipments.csv` (20 records realistic) | `demo_esg_data.csv` (100+ records) | ☐ Realistic demo data |
| 4.9 | **Supplier profiles** | `demo_suppliers.yaml` (20 suppliers) | Company profiles YAML | ☐ YAML format for profiles |
| 4.10 | **Data quality tiers** | High/Medium/Low classification | Same | ☐ Quality scoring built in |

**Data Completeness:** If reference data <90% complete, calculations will fail.

---

### ✅ CATEGORY 5: PIPELINE ORCHESTRATION

**Principle:** CBAM's `cbam_pipeline.py` orchestrates agents. CSRD does the same, scaled up.

| # | Item | CBAM Pattern | CSRD Implementation | Verification |
|---|------|--------------|-------------------|--------------|
| 5.1 | **Main pipeline file** | `cbam_pipeline.py` (single orchestrator) | `csrd_pipeline.py` | ☐ Single orchestrator file |
| 5.2 | **Pipeline class** | `class CBAMPipeline` | `class CSRDPipeline` | ☐ Same class pattern |
| 5.3 | **Sequential agent calls** | Agent1 → Agent2 → Agent3 | Agent1 → Agent2 → ... → Agent6 | ☐ Sequential, not parallel (for audit trail) |
| 5.4 | **Intermediate outputs** | Saves JSON between stages | Same | ☐ All intermediate JSON saved |
| 5.5 | **Error handling** | Pipeline fails fast on errors | Same | ☐ Fail fast, no silent errors |
| 5.6 | **Progress reporting** | Logs progress % for each stage | Same | ☐ User sees progress |
| 5.7 | **Performance tracking** | Tracks time per agent | Same | ☐ Pipeline emits perf metrics |
| 5.8 | **Configuration loading** | Loads all paths at init | Same | ☐ Config-driven, not hardcoded paths |
| 5.9 | **Output packaging** | Creates final report + summary | Creates XBRL + PDF + JSON | ☐ Multi-format output |
| 5.10 | **Provenance assembly** | Combines all agent provenance | Same | ☐ Complete lineage from input → output |

**Integration Test:** Full pipeline MUST run end-to-end before release.

---

### ✅ CATEGORY 6: PROVENANCE & AUDIT TRAIL

**Principle:** CRITICAL for regulatory use. CBAM tracks everything; CSRD MUST do the same.

| # | Item | CBAM Approach | CSRD Requirement | Verification |
|---|------|--------------|-----------------|--------------|
| 6.1 | **Provenance module** | `provenance/provenance_utils.py` | EXACT same module | ☐ Copy CBAM provenance code |
| 6.2 | **SHA256 hashing** | Every intermediate output hashed | Same | ☐ Hashes in all outputs |
| 6.3 | **Calculation lineage** | Tracks `input → operation → output` | Same | ☐ Every calculation documented |
| 6.4 | **Data source tracking** | Tracks which file/supplier provided data | Same | ☐ Data lineage to source |
| 6.5 | **Agent version tracking** | Records agent version in provenance | Same | ☐ Version stamps |
| 6.6 | **Environment info** | Python version, OS, timestamp | Same | ☐ Reproducibility metadata |
| 6.7 | **Manifest files** | `manifest.json` with all provenance | Same | ☐ Machine-readable manifest |
| 6.8 | **Human-readable audit** | Markdown audit trail | Same + PDF for auditors | ☐ Auditor package |
| 6.9 | **Provenance API** | Functions: `create_record()`, `verify()`, `chain()` | Same | ☐ Consistent provenance API |
| 6.10 | **Third-party verification** | Designed for external audit | Same | ☐ Audit-ready format |

**Regulatory Requirement:** Without complete provenance, CSRD reports cannot be assured.

---

### ✅ CATEGORY 7: TESTING STRATEGY

**Principle:** CBAM has 65+ tests. CSRD needs 150+ tests (2-3× scale).

| # | Item | CBAM Testing | CSRD Target | Verification |
|---|------|--------------|-------------|--------------|
| 7.1 | **Unit test coverage** | Target: 80%+ (not yet reached in CBAM) | Target: 85%+ | ☐ Pytest coverage report >85% |
| 7.2 | **Calculator tests** | 100% coverage (CRITICAL) | 100% coverage | ☐ ALL formulas tested |
| 7.3 | **Test per agent** | `test_shipment_intake_agent.py` (one file per agent) | Same pattern × 6 | ☐ 6 test files for 6 agents |
| 7.4 | **Integration tests** | `test_pipeline_integration.py` | Same | ☐ End-to-end pipeline tests |
| 7.5 | **Test data generators** | `generate_demo_shipments.py` | Similar for ESRS data | ☐ Synthetic data generators |
| 7.6 | **Pytest fixtures** | `tests/conftest.py` with shared fixtures | Same | ☐ Reusable test fixtures |
| 7.7 | **Mock LLM responses** | N/A (no LLM in CBAM) | Mock for MaterialityAgent | ☐ Deterministic test for AI agent |
| 7.8 | **Performance tests** | `scripts/benchmark.py` | Same | ☐ Benchmarks for all agents |
| 7.9 | **Schema validation tests** | Tests that bad data fails validation | Same | ☐ Negative test cases |
| 7.10 | **CLI tests** | `test_cli.py` | Same | ☐ CLI commands tested |
| 7.11 | **SDK tests** | `test_sdk.py` | Same | ☐ SDK functions tested |
| 7.12 | **Provenance tests** | `test_provenance.py` | Same | ☐ Provenance correctness verified |

**Quality Gate:** <85% coverage → DO NOT RELEASE.

---

### ✅ CATEGORY 8: PERFORMANCE TARGETS

**Principle:** CBAM achieves <10 min for 10K records. CSRD targets <30 min for 10K data points.

| # | Item | CBAM Benchmark | CSRD Target | Verification |
|---|------|----------------|-------------|--------------|
| 8.1 | **End-to-end time** | <10 min for 10K shipments | <30 min for 10K data points | ☐ Full pipeline benchmark |
| 8.2 | **IntakeAgent throughput** | 1,000+ records/sec | 1,000+ records/sec | ☐ Benchmark script |
| 8.3 | **CalculatorAgent latency** | <3 ms per shipment | <5 ms per metric | ☐ Per-metric timing |
| 8.4 | **ReportingAgent time** | <1 sec for 10K | <5 min (XBRL complexity) | ☐ XBRL generation timed |
| 8.5 | **Memory footprint** | <500 MB for 100K shipments | <1 GB for 100K data points | ☐ Memory profiling |
| 8.6 | **Scalability test** | Tested up to 100K records | Test up to 100K data points | ☐ Scalability benchmarks |
| 8.7 | **Performance regression** | Track perf over time | Same | ☐ CI/CD performance tests |

**Failure Condition:** If >3× slower than targets, architecture needs rework.

---

### ✅ CATEGORY 9: CLI & SDK

**Principle:** CBAM provides both CLI and Python SDK. CSRD MUST match.

| # | Item | CBAM Pattern | CSRD Implementation | Verification |
|---|------|--------------|-------------------|--------------|
| 9.1 | **CLI entry point** | `cli/cbam_commands.py` | `cli/csrd_commands.py` | ☐ Same Click-based CLI |
| 9.2 | **Main command** | `python cbam_pipeline.py --input ...` | `python csrd_pipeline.py --input ...` | ☐ Simple CLI invocation |
| 9.3 | **CLI options** | `--input`, `--output`, `--summary`, etc. | Same + CSRD-specific | ☐ Consistent option naming |
| 9.4 | **Help text** | Comprehensive `--help` | Same | ☐ All options documented |
| 9.5 | **SDK one-function API** | `cbam_build_report()` | `csrd_build_report()` | ☐ Dead-simple SDK entry |
| 9.6 | **SDK returns dataclass** | `CBAMReport` dataclass | `CSRDReport` dataclass | ☐ Type-safe returns |
| 9.7 | **SDK examples** | `examples/quick_start_sdk.py` | Same | ☐ SDK examples work |
| 9.8 | **Jupyter support** | Can use in notebooks | Same | ☐ DataFrame in/out support |

**Usability Test:** Non-technical user MUST be able to run CLI with <5 min training.

---

### ✅ CATEGORY 10: DOCUMENTATION

**Principle:** CBAM has comprehensive docs. CSRD must match.

| # | Item | CBAM Documentation | CSRD Requirement | Verification |
|---|------|-------------------|-----------------|--------------|
| 10.1 | **README.md** | 647 lines, comprehensive quick start | Similar depth | ☐ <10 min to first run |
| 10.2 | **PROJECT_CHARTER.md** | Clear mission, scope, success criteria | Exists | ☐ Project alignment |
| 10.3 | **BUILD_JOURNEY.md** | Detailed build narrative | Same | ☐ Development story |
| 10.4 | **API_REFERENCE.md** | Function signatures, examples | Same | ☐ Complete API docs |
| 10.5 | **USER_GUIDE.md** | Step-by-step guide | Same | ☐ User-friendly |
| 10.6 | **DEPLOYMENT_GUIDE.md** | Production deployment | Same | ☐ Ops-ready |
| 10.7 | **TROUBLESHOOTING.md** | Common issues + fixes | Same | ☐ Self-service support |
| 10.8 | **Agent specs** | YAML specs for each agent | 6 specs | ☐ Specs match implementation |
| 10.9 | **Inline docstrings** | All functions documented | Same | ☐ Pydoc-ready |
| 10.10 | **Example scripts** | Working examples | Same | ☐ Examples execute without errors |

**Documentation Test:** External reviewer MUST be able to run pipeline from docs alone.

---

### ✅ CATEGORY 11: REGULATORY COMPLIANCE

**Principle:** CBAM complies with EU CBAM 2023/956. CSRD complies with EU CSRD 2023/2464.

| # | Item | CBAM Compliance | CSRD Compliance | Verification |
|---|------|-----------------|----------------|--------------|
| 11.1 | **Regulation referenced** | EU CBAM 2023/956 in README | EU CSRD 2023/2464 | ☐ Regulation number in docs |
| 11.2 | **Compliance rules** | 50+ rules from regulation | 200+ rules from ESRS | ☐ Rules map to regulation articles |
| 11.3 | **Output format** | EU Registry JSON format | ESEF XBRL format | ☐ Format validator passes |
| 11.4 | **Data completeness** | Enforces mandatory fields | Same | ☐ Missing data = ERROR |
| 11.5 | **Validation engine** | Checks all compliance rules | Same | ☐ Rule engine comprehensive |
| 11.6 | **Audit package** | Generates auditor docs | Same + third-party assurance | ☐ Auditor-ready output |
| 11.7 | **Disclaimer** | "Not legal advice" | Same | ☐ Legal disclaimer |
| 11.8 | **Data sources cited** | IEA, IPCC, etc. | EFRAG, ISSB, etc. | ☐ All factors sourced |
| 11.9 | **Methodology documented** | Calculation methods explained | Same | ☐ Methodology transparency |
| 11.10 | **Multi-language support** | Not in v1.0 | Planned for v1.1 | ☐ Roadmap item |

**Regulatory Risk:** If output format doesn't match ESEF, reports CANNOT be filed.

---

### ✅ CATEGORY 12: GREENLANG INTEGRATION

**Principle:** Both apps are GreenLang packs. Same integration level required.

| # | Item | CBAM Integration | CSRD Requirement | Verification |
|---|------|-----------------|-----------------|--------------|
| 12.1 | **pack.yaml structure** | Complete pack definition | Mirror structure | ☐ Validates with GL CLI |
| 12.2 | **gl.yaml metadata** | GreenLang metadata | Same | ☐ Same fields as CBAM |
| 12.3 | **Agent definitions in pack** | 3 agents defined | 6 agents defined | ☐ All agents in pack.yaml |
| 12.4 | **Tool definitions** | Tools listed per agent | Same | ☐ Clear tool taxonomy |
| 12.5 | **GreenLang version** | `>=0.3.0` | Same | ☐ Version compatibility |
| 12.6 | **Hub publication status** | `draft` initially | Same progression | ☐ Publication workflow |
| 12.7 | **Tags and categories** | Compliance, reporting, etc. | Same + CSRD-specific | ☐ Searchable tags |
| 12.8 | **Dependencies** | All deps in pack.yaml | Same | ☐ Complete dependency list |
| 12.9 | **Examples** | Demo data paths in pack | Same | ☐ Pack includes examples |
| 12.10 | **README linked in pack** | Pack points to README | Same | ☐ Documentation links |

**GreenLang Test:** Pack MUST install with `gl install` command.

---

## 🚨 CRITICAL FAILURE MODES (RED FLAGS)

These indicate the project is OFF TRACK and needs immediate correction:

### ❌ FAILURE MODE 1: Zero-Hallucination Violation
**Symptom:** LLM is used in CalculatorAgent or any deterministic path
**Impact:** CRITICAL - Regulatory failure, audit failure
**Fix:** IMMEDIATELY remove LLM, use database lookups only
**CBAM Precedent:** EmissionsCalculatorAgent has ZERO LLM calls

### ❌ FAILURE MODE 2: No Provenance
**Symptom:** Cannot trace calculation back to source data
**Impact:** CRITICAL - Third-party assurance impossible
**Fix:** Implement `provenance_utils.py` from CBAM
**CBAM Precedent:** Every calculation has SHA256 hash chain

### ❌ FAILURE MODE 3: Monolithic Agents
**Symptom:** One huge file instead of 6 separate agents
**Impact:** HIGH - Unmaintainable, untestable
**Fix:** Split into 6 files following CBAM pattern
**CBAM Precedent:** 3 agents in 3 files, <1,500 lines each

### ❌ FAILURE MODE 4: No Validation Rules
**Symptom:** Rules hardcoded in Python instead of YAML
**Impact:** HIGH - Cannot update rules without code changes
**Fix:** Extract to `esrs_compliance_rules.yaml`
**CBAM Precedent:** `cbam_rules.yaml` with 50+ rules

### ❌ FAILURE MODE 5: Missing Tests
**Symptom:** <70% test coverage, especially in CalculatorAgent
**Impact:** HIGH - Production bugs, regulatory risk
**Fix:** Write tests to 100% for calculations
**CBAM Precedent:** Target is 80%+ coverage

### ❌ FAILURE MODE 6: Performance Regression
**Symptom:** >3× slower than targets (e.g., >90 min for 10K points)
**Impact:** MEDIUM - User frustration, scalability issues
**Fix:** Profile and optimize bottlenecks
**CBAM Precedent:** <10 min for 10K shipments

### ❌ FAILURE MODE 7: Wrong Output Format
**Symptom:** JSON instead of XBRL, or XBRL that fails validation
**Impact:** CRITICAL - Reports cannot be filed
**Fix:** Use Arelle for XBRL, validate against ESEF schema
**CBAM Precedent:** JSON for EU Registry (correct format)

### ❌ FAILURE MODE 8: Incomplete Documentation
**Symptom:** README doesn't allow <10 min quick start
**Impact:** MEDIUM - Adoption failure, support burden
**Fix:** Expand README to CBAM's level (647 lines)
**CBAM Precedent:** Comprehensive README with examples

---

## 📋 DEVELOPMENT PHASE CHECKLIST

Use this to verify each phase is complete before moving to the next.

### ✅ PHASE 1: FOUNDATION (Days 1-3)
- [ ] Directory structure mirrors CBAM (20/20 items from Category 1)
- [ ] `requirements.txt` includes all CBAM dependencies + XBRL/AI
- [ ] `provenance_utils.py` copied from CBAM
- [ ] `setup.py` matches CBAM pattern
- [ ] `.env.example` created
- [ ] README.md started

**Gate:** Cannot proceed without complete structure.

### ✅ PHASE 2: AGENTS (Days 4-18)
- [ ] IntakeAgent: 90%+ test coverage
- [ ] CalculatorAgent: 100% test coverage (CRITICAL)
- [ ] MaterialityAgent: 80%+ test coverage, LLM mocked
- [ ] AggregatorAgent: 90%+ test coverage
- [ ] ReportingAgent: 85%+ test coverage
- [ ] AuditAgent: 95%+ test coverage
- [ ] All agents follow class pattern from Category 3
- [ ] Zero-hallucination verified (12/12 items from Category 2)

**Gate:** Each agent MUST pass tests before pipeline integration.

### ✅ PHASE 3: PIPELINE (Days 19-21)
- [ ] `csrd_pipeline.py` orchestrates all 6 agents
- [ ] Sequential execution (not parallel)
- [ ] Intermediate outputs saved as JSON
- [ ] Provenance chain assembled (10/10 items from Category 6)
- [ ] Performance <30 min for 10K data points

**Gate:** Full pipeline integration test passes.

### ✅ PHASE 4: CLI & SDK (Days 22-27)
- [ ] CLI matches CBAM pattern (8/8 items from Category 9)
- [ ] `csrd_build_report()` one-function API works
- [ ] Examples execute without errors
- [ ] Help text comprehensive

**Gate:** Non-technical user can run CLI.

### ✅ PHASE 5: TESTING (Days 28-32)
- [ ] Overall test coverage >85%
- [ ] Calculator test coverage = 100%
- [ ] Integration tests pass
- [ ] Performance benchmarks meet targets (7/7 items from Category 8)

**Gate:** <85% coverage → DO NOT RELEASE.

### ✅ PHASE 6: DOCUMENTATION (Days 36-38)
- [ ] README.md matches CBAM depth (10/10 items from Category 10)
- [ ] All agent specs up-to-date
- [ ] API reference complete
- [ ] User guide comprehensive

**Gate:** External reviewer can run pipeline from docs.

### ✅ PHASE 7: REGULATORY (Days 39-42)
- [ ] 200+ compliance rules implemented
- [ ] XBRL output validates against ESEF schema
- [ ] Audit package generated
- [ ] All data sources cited (10/10 items from Category 11)

**Gate:** Regulatory expert review passes.

### ✅ PHASE 8: GREENLANG (Final)
- [ ] pack.yaml complete (10/10 items from Category 12)
- [ ] gl.yaml complete
- [ ] Pack installs with `gl install`
- [ ] Hub publication ready

**Gate:** GreenLang Hub acceptance.

---

## 🎯 CRITICAL SUCCESS FACTORS (From CBAM)

These are the **non-negotiable** elements that made CBAM successful:

### 1. **Tool-First, Always**
- Database lookups > LLM guesses
- Python arithmetic > AI estimation
- Deterministic > Probabilistic (for calculations)

### 2. **Complete Provenance**
- SHA256 hashes everywhere
- Data lineage from source → output
- Reproducibility guaranteed

### 3. **Agent Separation**
- One agent = one job
- Clear inputs/outputs
- Independent testing

### 4. **Data as Code**
- CN codes, factors, rules in files
- Not hardcoded in Python
- Version controlled

### 5. **Validation-First**
- 50+ rules in CBAM → 200+ in CSRD
- Fail fast on bad data
- Error codes for every issue

### 6. **Performance Obsession**
- Benchmark every agent
- <10 min end-to-end (CBAM) → <30 min (CSRD)
- Memory profiling

### 7. **Documentation Excellence**
- <10 min quick start
- Comprehensive examples
- Clear agent specs

### 8. **Test Everything**
- 80%+ coverage minimum
- 100% for calculations
- Integration tests mandatory

---

## 🔍 WHAT TO WATCH OUT FOR

### Common Pitfalls (Based on CBAM Experience)

1. **Scope Creep**
   - CBAM stayed focused on Transitional Period
   - CSRD should stay focused on ESRS reporting
   - Don't add "AI insights dashboard" in v1.0

2. **Overuse of AI**
   - CBAM uses ZERO AI (intentional)
   - CSRD should use AI ONLY in MaterialityAgent
   - Temptation to "add AI" to calculations → RESIST

3. **Insufficient Testing**
   - CBAM target is 80%, not yet reached
   - CSRD MUST hit 85% before release
   - Calculator MUST be 100%

4. **Poor Documentation**
   - CBAM has 647-line README
   - CSRD needs similar depth
   - Examples MUST work out-of-box

5. **Wrong Output Format**
   - CBAM outputs JSON (correct for EU Registry)
   - CSRD MUST output ESEF XBRL (different)
   - Validate format early

6. **Hardcoding**
   - CBAM puts data in files (good)
   - Don't put ESRS factors in code
   - Rules MUST be in YAML

7. **Performance Neglect**
   - CBAM benchmarks continuously
   - CSRD has 3× more data
   - Profile early, optimize often

---

## 📊 FINAL VERIFICATION MATRIX

Before declaring CSRD "complete," verify ALL categories:

| Category | Items | Min Pass Rate | CSRD Status |
|----------|-------|--------------|-------------|
| 1. Project Structure | 20 | 20/20 (100%) | ☐ |
| 2. Zero-Hallucination | 10 | 10/10 (100%) | ☐ |
| 3. Agent Architecture | 12 | 11/12 (92%) | ☐ |
| 4. Data & Validation | 10 | 9/10 (90%) | ☐ |
| 5. Pipeline Orchestration | 10 | 10/10 (100%) | ☐ |
| 6. Provenance & Audit | 10 | 10/10 (100%) | ☐ |
| 7. Testing Strategy | 12 | 11/12 (92%) | ☐ |
| 8. Performance Targets | 7 | 7/7 (100%) | ☐ |
| 9. CLI & SDK | 8 | 8/8 (100%) | ☐ |
| 10. Documentation | 10 | 9/10 (90%) | ☐ |
| 11. Regulatory Compliance | 10 | 10/10 (100%) | ☐ |
| 12. GreenLang Integration | 10 | 10/10 (100%) | ☐ |

**OVERALL PASS THRESHOLD: 115/119 items (97%)**

---

## 🚀 CONCLUSION

### Summary for Development Team

**CSRD is CBAM scaled 2-3×:**
- Same architecture (6 agents vs 3)
- Same zero-hallucination principle
- Same provenance system
- Same testing rigor
- Same documentation quality
- 4× more validation rules
- 70× more data points
- Adds XBRL complexity

**USE THIS CHECKLIST TO:**
1. Verify each development phase
2. Catch deviations early
3. Ensure regulatory compliance
4. Maintain CBAM's quality standard
5. Avoid common pitfalls

**REMEMBER:**
- CBAM succeeded because of discipline: tool-first, provenance-always, test-everything
- CSRD MUST follow the same discipline
- When in doubt, check CBAM's implementation
- Better to copy CBAM exactly than to "improve" and break

**FINAL WORD:**
If CSRD passes 115/119 items (97%), it will be production-ready and regulatory-compliant. Anything less is unacceptable.

---

**Document End**

*Use this checklist in daily standups, code reviews, and release gates.*
