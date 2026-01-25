# CBAM IMPORTER COPILOT - PROJECT CHARTER

## Mission Statement

Build a working demo of CBAM (Carbon Border Adjustment Mechanism) Importer Copilot that generates EU CBAM Transitional Registry filing packs in under 10 minutes using synthetic data.

## Problem Statement

EU importers face complex compliance requirements for CBAM transitional reporting (Q4 2025 due Jan 31, 2026):
- Manual data entry is error-prone and time-consuming
- Emission calculations require specialized knowledge
- Complex rules (20% cap for complex goods, default value restrictions)
- Multiple data sources to reconcile (customs, suppliers, EU defaults)

## Solution Overview

**CBAM Importer Copilot** automates the entire workflow:
1. **Input:** Shipment data (CSV/XLSX) + Supplier registry (YAML)
2. **Processing:** 3 AI agents orchestrate calculation and validation
3. **Output:** Filing pack (CSV for portal upload, PDF summary, provenance manifest)

## Success Criteria

### Functional Requirements (MUST HAVE)
- ✅ CLI command: `gl cbam report --demo` generates filing pack
- ✅ Python SDK: `cbam_build_report()` callable from code
- ✅ 3 AI agents operational (ShipmentIntake, EmissionsCalculator, ReportingPackager)
- ✅ Synthetic data realistic enough for investor/customer demos
- ✅ Output files: registry_upload.csv, summary.pdf, manifest.json, metrics.json
- ✅ Provenance: Signed manifests with Sigstore
- ✅ Tests: 65+ unit tests, integration tests passing
- ✅ Documentation: 10-minute quickstart, API reference

### Performance Targets
- ⚡ **TTV:** <10 minutes from install to first filing pack
- ⚡ **Processing:** <3 seconds per 1,000 shipments
- ⚡ **Accuracy:** 100% calculation correctness (zero hallucinated numbers)
- ⚡ **Coverage:** >80% test coverage

### Business Goals
- 🎯 Published to GreenLang Hub as "cbam-importer-demo"
- 🎯 Showcase for investor/customer demos
- 🎯 Validate Agent Factory for domain-specific applications
- 🎯 Prove GreenLang infrastructure readiness

## Approach: SYNTHETIC-FIRST STRATEGY

**Phase 1 (Days 1-2): Synthetic Data Foundation**
- Build emission factors from public sources (IEA, IPCC, World Steel Association)
- Extract CN codes from CBAM Regulation Annex I
- Generate 500 realistic demo shipments
- Create 20 demo supplier profiles

**Phase 2 (Day 2): Schemas & Rules**
- Define JSON schemas for Shipment, Supplier, Registry outputs
- Create CBAM rules specification (YAML)
- Implement validator functions

**Phase 3 (Day 2): Agent Specifications**
- Write detailed specs for 3 agents with tool definitions
- Prepare for Agent Factory generation

**Phase 4 (Days 3-4): Agent Generation**
- Use Agent Factory to generate 3 AI agents
- Write 65+ unit tests
- Integration testing

**Phase 5-10 (Days 5-10): Integration, CLI, SDK, Docs, Launch**
- Assemble pack structure
- Build CLI and Python SDK
- Add provenance and observability
- Write documentation
- Validation and Hub publication

## Timeline

- **Day 0:** Setup (4 hours) ✅ COMPLETE
- **Days 1-2:** Synthetic data + schemas (12 hours)
- **Days 3-4:** Agents (12 hours)
- **Days 5-6:** CLI/SDK (12 hours)
- **Days 7-8:** Docs + validation (12 hours)
- **Days 9-10:** Launch prep (8 hours)

**Total: 10 working days (60 hours)**

## Key Design Principles

1. **Tool-First Architecture:** All numeric calculations use deterministic tools (ZERO hallucinated numbers)
2. **Demo Mode:** Clearly labeled as synthetic data for illustration
3. **Production Path:** Design allows easy swap to real EU data later
4. **Leverage Existing Infrastructure:** Use Agent Factory, Pack system, CLI framework at 76.4% complete
5. **Ship Fast, Iterate:** MVP now, production version later

## Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Agent Factory issues | 20% | +18 hrs | Fallback to manual build with reference agents |
| Synthetic data quality | 10% | +4 hrs | Cross-reference multiple public sources |
| Integration bugs | 30% | +4 hrs | Buffer time in validation phase |
| Scope creep | 40% | Timeline slip | Strict scope control, defer non-essentials |

## Out of Scope (For This MVP)

- ❌ Real EU Commission default values (demo only)
- ❌ Actual CBAM portal uploads (validation only)
- ❌ Production-grade error recovery
- ❌ Multi-language support
- ❌ Advanced analytics/dashboards
- ❌ Pilot customer onboarding

## Stakeholders

- **Lead Engineer:** Full-time execution
- **PM/Product:** Scope control, launch coordination
- **Head of AI (Claude):** Architecture guidance, code review
- **Mentor:** Strategic feedback

## Version History

- **v0.1.0** (Target: Day 10): Demo release to GreenLang Hub
- **v0.2.0** (Future): Production mode with real EU data
- **v1.0.0** (Future): Full production release with pilot validation

---

**Status:** IN PROGRESS - Day 0 Complete, Starting Day 1 (Synthetic Data)

**Last Updated:** 2025-10-15 (Project Start)
