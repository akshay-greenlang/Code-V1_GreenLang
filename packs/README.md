# GreenLang Solution Packs

Solution Packs are curated, deployable bundles that combine GreenLang's AI agents, applications, configurations, and workflows into ready-to-use compliance solutions.

## Architecture

```
Packs (deployable solutions)
  └── Applications (single-regulation UIs)
       └── Agents (calculation & processing engines)
            └── Infrastructure (database, cache, auth, observability)
```

Each pack **references and orchestrates** components from the `greenlang/` agent layer and `applications/` app layer — it does not duplicate code.

## Pack Categories

### EU Compliance Packs
| Pack | Name | Tier | Agents | Tests | Status |
|------|------|------|--------|-------|--------|
| PACK-001 | CSRD Starter Pack | Starter | 66+ | 123 | **BUILT** |
| PACK-002 | CSRD Professional Pack | Professional | 93+ | 313 | **BUILT** |
| PACK-003 | CSRD Enterprise Pack | Enterprise | 135+ | 355 | **BUILT** |
| PACK-004 | CBAM Readiness Pack | Standalone | 45+ | 268 | **BUILT** |
| PACK-005 | CBAM Complete Pack | Complete | 45+ | 367 | **BUILT** |
| PACK-006 | EUDR Compliance Pack | TBD | TBD | TBD | Planned |
| PACK-007 | EU Taxonomy Pack | TBD | TBD | TBD | Planned |

### US Climate Disclosure Packs
| Pack | Name | Status | Description |
|------|------|--------|-------------|
| PACK-010 | SB-253 Compliance Pack | Planned | California Climate Corporate Data Accountability Act |
| PACK-011 | SEC Climate Pack | Planned | SEC Climate-Related Disclosures |

### Voluntary Standards Packs
| Pack | Name | Status | Description |
|------|------|--------|-------------|
| PACK-020 | GHG Protocol Pack | Planned | Full GHG Protocol Corporate Standard reporting |
| PACK-021 | CDP Disclosure Pack | Planned | CDP Climate Change Questionnaire |
| PACK-022 | SBTi Target Pack | Planned | Science-Based Targets Initiative validation |
| PACK-023 | TCFD Reporting Pack | Planned | Task Force on Climate-Related Financial Disclosures |

### Industry-Specific Packs
| Pack | Name | Status | Description |
|------|------|--------|-------------|
| PACK-030 | ISO 14064 Pack | Planned | ISO 14064 organizational GHG inventory |
| PACK-031 | Product Carbon Footprint Pack | Planned | Product-level LCA and carbon footprinting |

## Pack Structure

Each pack follows a standard structure:

```
PACK-XXX-name/
├── pack.yaml              # Pack manifest (components, dependencies, version)
├── README.md              # Pack documentation
├── workflows/             # Pre-built orchestration workflows (DAG definitions)
├── config/                # Pack-specific configuration & environment overrides
├── templates/             # Report templates, dashboard configs, email templates
├── integrations/          # Pack-level integration layer (agent wiring, data flows)
└── tests/                 # Pack-level integration & E2E tests
```

### pack.yaml Manifest

The `pack.yaml` manifest declaratively defines:
- **metadata**: Name, version, description, category, regulatory references
- **components**: Which agents, apps, and migrations are included
- **workflows**: Pre-configured orchestration DAGs
- **config**: Default configuration with environment overrides
- **dependencies**: Required infrastructure and external services

## Installation

```bash
# Install a pack
greenlang pack install eu-compliance/PACK-001-csrd-starter

# List available packs
greenlang pack list

# Verify pack health
greenlang pack verify PACK-001
```

## Development

See each pack's README for development instructions and contribution guidelines.
