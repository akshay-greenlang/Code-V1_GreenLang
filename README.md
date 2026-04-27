# GreenLang

**GreenLang Climate OS: an auditable climate data foundation and compliance application stack, with GreenLang Factors as the current hardened foundation.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.3.0-green.svg)](https://github.com/akshay-greenlang/Code-V1_GreenLang/releases)

## What This Repo Is

GreenLang is being built as a climate operating system for regulated enterprises: a shared substrate for emission factors, activity data, evidence, policy logic, and auditable calculations, with applications such as CBAM, CSRD/Comply, and Scope Engine built on top.

The active release focus is **GreenLang Factors v0.1 Alpha**. The broader Climate OS modules remain in the repository for roadmap and integration work, but they are not all release-supported surfaces yet.

## Current Product Status

| Area | Current status |
|---|---|
| **GreenLang Factors** | v0.1 Alpha foundation integrated. Scope 1/2 factor catalog, source registry, provenance gates, source-rights enforcement, read-only API, Python SDK, governance docs, and alpha tests are active release surfaces. |
| **CBAM, Comply, Scope Engine** | Roadmap/partial implementation. These applications build on the same substrate but are not represented here as fully supported GA products unless their own CI and release manifests say so. |
| **Agent Runtime, Policy Graph, Ledger, Evidence Vault, Connect** | Platform roadmap components. Code and stubs exist, but active release gating is milestone-specific. |
| **Archive and planning folders** | Historical or future-scope material. They are intentionally excluded from default release gates until promoted by a release profile. |

## Architecture

GreenLang is organized into four layers:

```text
L4 Applications      CBAM, Comply, Scope Engine, future sector apps
L3 Intelligence      Policy Graph, Agent Runtime, evaluation harnesses
L2 System of Record  Climate Ledger, Evidence Vault, audit bundles
L1 Foundation        Factors, Connect, Entity Graph, IoT/data schemas
```

The immediate hardened path is L1 **Factors**:

- Canonical source registry and source-rights matrix.
- Read-only v0.1 API surface.
- Python SDK package surface for alpha partners.
- Provenance fields on alpha factor records.
- Release-profile gating so future functionality does not leak into alpha.

## Quick Start

```bash
git clone https://github.com/akshay-greenlang/Code-V1_GreenLang.git
cd Code-V1_GreenLang
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -e ".[factors-test]"
python -m pytest tests/factors/v0_1_alpha -q
```

## Factors v0.1 Alpha API

The alpha public API is intentionally read-only:

- `GET /v1/healthz`
- `GET /v1/factors`
- `GET /v1/factors/{urn}`
- `GET /v1/sources`
- `GET /v1/packs`

Future surfaces such as GraphQL, SQL-over-HTTP, billing, OEM distribution, TypeScript SDK publishing, mutation flows, and public commercial packs are controlled by release profile and are not part of the v0.1 Alpha public contract.

## Development

Common checks:

```bash
python -m pytest tests/factors/v0_1_alpha -q
python -m ruff check greenlang/factors tests/factors/v0_1_alpha --select E,F,W --ignore E501
python -m pip install --dry-run -e ".[test,dev,factors-test]"
```

For broader repository orientation, see:

- `docs/factors/product/MASTER_PRD.md`
- `docs/factors/engineering/ENGINEERING_CHARTER.md`
- `docs/factors/governance/SOURCE_OF_TRUTH_GOVERNANCE.md`
- `docs/factors/roadmap/SOURCE_OF_TRUTH_MANIFEST.md`

## Repository Governance

`master` is the supportable integration branch. Active CI/security gates are scoped to release-supported code first. Archive, planning, and future-scope directories remain available for reference but should not block the current release unless they are explicitly promoted into an active milestone.

Historical tags are immutable release pointers and are preserved as-is.

## License And Contact

- **License:** Apache 2.0, see `LICENSE`.
- **Issues:** https://github.com/akshay-greenlang/Code-V1_GreenLang/issues
- **Security:** security@greenlang.io
- **Homepage:** https://greenlang.io

**Version:** 0.3.0
