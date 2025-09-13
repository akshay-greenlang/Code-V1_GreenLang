# GreenLang Agent Governance Framework

## Category 1: Core Governance & Code Quality Agents

### Existing Agents
1. **GL-SpecGuardian** - Validates GreenLang spec files (pack.yaml, gl.yaml, run.json)
2. **GL-CodeSentinel** - Code health, linting, type checking, style compliance
3. **GL-SecScan** - Security scanning for secrets, CVEs, policy violations
4. **GL-PolicyLinter** - OPA Rego policy auditing for security compliance

### Status: Complete ✓

## Category 2: Runtime, Packs, Data & Connectors Agents

### Existing Agents
1. **GL-DeterminismAuditor** - Verifies reproducible behavior across runs

### New Agents Needed
2. **GL-PackQC** - Pack quality control and validation
3. **GL-DataFlowGuardian** - Data lineage and flow validation
4. **GL-ConnectorValidator** - Connector security and compatibility checks

### Status: In Progress

## Category 3: Hub, Distribution, Partners Agents

### New Agents Needed
1. **GL-HubRegistrar** - Hub registry validation and publishing checks
2. **GL-PartnerCompliance** - Partner integration and compliance validation
3. **GL-DistributionValidator** - Distribution channel security and integrity

### Status: To Be Created

## Category 4: Release Train & Exit Bars Agents

### Existing Agents
1. **GL-SupplyChainSentinel** - SBOM, signatures, provenance validation

### New Agents Needed
2. **GL-ExitBarAuditor** - Release criteria and exit bar validation
3. **GL-ReleaseManifestValidator** - Release manifest completeness and accuracy

### Status: In Progress

## Category 5: Developer Experience & Safety Nets Agents

### Existing Agents
1. **Project-Status-Reporter** - Stakeholder reports and progress tracking

### New Agents Needed
2. **GL-DevExHelper** - Developer experience improvements and helpful suggestions
3. **GL-SafetyNetMonitor** - Runtime safety checks and resource monitoring

### Status: In Progress

## Integration Strategy

Each agent category serves a specific governance purpose:
- **Core Governance**: Ensures code quality and security fundamentals
- **Runtime & Data**: Validates execution consistency and data integrity
- **Hub & Distribution**: Manages ecosystem and partner interactions
- **Release Train**: Enforces production readiness criteria
- **Developer Experience**: Improves usability and provides safety nets