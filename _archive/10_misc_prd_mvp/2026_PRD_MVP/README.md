# GreenLang MVP 2026 - Product Requirements

This folder contains the complete Product Requirements Document (PRD) for the GreenLang MVP: **CBAM Compliance Pack + Pack Runner**.

## Contents

| File | Description |
|------|-------------|
| `GreenLang_PRD_MVP_2026.md` | Complete PRD with all specifications |
| `README.md` | This file |
| `DECISION_LOG.md` | Record of key decisions made during PRD development |
| `AGENT_MAPPING.md` | Mapping from 402-agent catalog to MVP agents |

## Quick Links

- **Executive Summary**: [Section 2](GreenLang_PRD_MVP_2026.md#2-executive-summary)
- **MVP Scope**: [Section 7](GreenLang_PRD_MVP_2026.md#7-mvp-scope-definition)
- **Functional Requirements**: [Section 8](GreenLang_PRD_MVP_2026.md#8-functional-requirements)
- **Agent Architecture**: [Section 11](GreenLang_PRD_MVP_2026.md#11-agent-architecture--pipeline)
- **Delivery Milestones**: [Section 22](GreenLang_PRD_MVP_2026.md#22-delivery-milestones)

## Key Decisions Summary

| Area | Decision |
|------|----------|
| Target Customer | Small/Mid EU Importers (10-100 lines/quarter) |
| Business Model | Open-source + paid support |
| Regulatory Scope | CBAM Transitional period only (2024-2025) |
| Product Categories | Steel & Iron + Aluminum |
| Interface | CLI-first (`gl run cbam --config ...`) |
| PDF Extraction | Excluded from MVP |
| Supplier Data | Defaults-first, optional override |
| Agent Chain | Minimal (7 agents) |
| Deployment | CLI + Docker Compose |
| Security | Local-first, no network |

## MVP Definition of Done

1. XSD-valid XML generated for EU Transitional Registry
2. Complete audit bundle (claims, lineage, assumptions, manifest)
3. Deterministic reruns (same inputs + versions = identical outputs)
4. Steel & Aluminum CN codes fully supported
5. Direct + indirect emissions calculated
6. Gap report generated
7. Error handling with actionable messages
8. 3+ golden dataset tests passing
9. Unit test coverage ≥80%
10. Regulatory expert review completed

## Next Steps

1. Review PRD with stakeholders
2. Assign owners to milestones
3. Begin M1 (Pack Spec + Input Template)
4. Set up regulatory expert review
