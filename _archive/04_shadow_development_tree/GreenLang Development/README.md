# GreenLang Climate OS - Development Hub

**Version:** 2.0 | **Last Updated:** February 2, 2026

---

## Overview

This is the consolidated **GreenLang Development** folder containing all organized code, documentation, and resources for building the GreenLang Climate Operating System.

---

## Folder Structure

```
GreenLang Development/
│
├── 01-Core-Platform/          # Runtime engine, SDK, core modules
│   ├── agents/                # Agent framework
│   ├── core/                  # Core logic
│   ├── data/                  # Data processing
│   ├── api/                   # API layer
│   └── cli/                   # Command line tools
│
├── 02-Applications/           # All GreenLang applications
│   ├── GL-CSRD-APP/          # CSRD compliance
│   ├── GL-CBAM-APP/          # CBAM compliance
│   ├── GL-EUDR-APP/          # EUDR compliance
│   ├── GL-VCCI-APP/          # Scope 3 intelligence
│   └── [50+ more apps]/
│
├── 03-Agents/                 # 402 canonical agents
│   ├── foundation/           # Layer 1: Foundation
│   ├── data/                 # Layer 2: Data connectors
│   ├── mrv/                  # Layer 3: MRV calculations
│   ├── planning/             # Layer 4: Decarbonization
│   └── [7 more layers]/
│
├── 04-Infrastructure/         # DevOps & infrastructure
│   ├── kubernetes/           # K8s manifests
│   ├── terraform/            # IaC
│   └── ci-cd/                # Pipelines
│
├── 05-Documentation/          # All documentation
│   ├── PRD/                  # Product requirements
│   ├── API/                  # API docs
│   ├── Agents/               # Agent specs
│   └── Guides/               # User guides
│
├── 06-Solution-Packs/         # Pre-packaged solutions
│   ├── compliance/           # Regulatory packs
│   ├── industry/             # Industry packs
│   └── energy/               # Energy packs
│
├── 07-Testing/                # Test infrastructure
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── e2e/                  # End-to-end tests
│
├── 08-Deployment/             # Deployment configs
│   ├── docker/               # Docker configs
│   ├── helm/                 # Helm charts
│   └── scripts/              # Deploy scripts
│
├── GreenLang_2026_PRD.md      # Master PRD document
├── DEVELOPMENT_STATUS.md      # Current status report
└── README.md                  # This file
```

---

## Quick Start

### 1. Development Environment Setup

```bash
# Clone the repository
cd GreenLang Development/01-Core-Platform

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your configurations

# Run tests
pytest tests/
```

### 2. Running Applications

```bash
# Start the API server
uvicorn greenlang.api:app --reload

# Or use Docker
docker-compose up -d
```

### 3. Agent Development

```bash
# Create new agent from template
python -m greenlang.cli.agent create --name MyAgent --layer mrv

# Run agent tests
python -m pytest tests/agents/test_my_agent.py
```

---

## Key Documents

| Document | Location | Description |
|----------|----------|-------------|
| **Master PRD** | `GreenLang_2026_PRD.md` | Complete 3-year roadmap |
| **Executive Summary** | `05-Documentation/01-EXECUTIVE-SUMMARY.md` | High-level overview |
| **Agent Specifications** | `05-Documentation/03-AGENT-SPECIFICATIONS.md` | All 402 agents |
| **Application Catalog** | `05-Documentation/05-APPLICATION-CATALOG.md` | 500+ applications |
| **3-Year Plan** | `05-Documentation/04-3-YEAR-DEVELOPMENT-PLAN.md` | Quarterly roadmap |

---

## Development Priorities

### Tier 1: CRITICAL (Now)
- [ ] GL-EUDR-APP - EU Deforestation (Dec 30, 2025)
- [ ] Agent Factory v1.0
- [ ] 10 Foundation Agents

### Tier 2: HIGH (Q1-Q2 2026)
- [ ] GL-SB253-APP - California SB 253
- [ ] Industrial decarbonization apps
- [ ] 200 agents total

### Tier 3: STRATEGIC (2026+)
- [ ] Building Performance Standards
- [ ] Supply Chain & Scope 3 enhanced
- [ ] Global expansion

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| API Framework | FastAPI |
| Database | PostgreSQL + TimescaleDB |
| Cache | Redis |
| Vector DB | pgvector |
| Container | Docker, Kubernetes |
| AI/ML | Claude API, PyTorch |
| Observability | OpenTelemetry, Grafana |

---

## External Integrations

### Ralphy Agent
Located at: `../ralphy-agent/`
Purpose: Task automation and project management

### Claude API
Purpose: Zero-hallucination LLM for climate intelligence

### Emission Factor Databases
- DEFRA (UK Government)
- EPA (US Environmental Protection Agency)
- Ecoinvent (Life Cycle Assessment)

---

## Team Resources

- **Slack:** #greenlang-dev
- **Jira:** greenlang.atlassian.net
- **Wiki:** docs.greenlang.io
- **GitHub:** github.com/greenlang

---

## Contact

For questions about this codebase:
- **Product:** product@greenlang.io
- **Engineering:** eng@greenlang.io
- **Support:** support@greenlang.io

---

*GreenLang Climate OS - Building the Operating System for a Net-Zero World*
